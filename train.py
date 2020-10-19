import os
import cv2
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from utils import dataset
from utils import transform as transform
from utils.util import poly_learning_rate
from utils.util import evaluate_single as evaluate
from model.model import Unet

cv2.ocl.setUseOpenCL(False)
cv2.setNumThreads(0)


def train_unet(train_loader, model, optimizer, epoch,args):

    model.train()
    max_iter = args.epochs * len(train_loader)
    for i, (input, label) in enumerate(train_loader):
        input = input.cuda()
        label = label.cuda()
        output, main_loss = model(input, label)
        loss = torch.mean(main_loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        label = label.view(label.shape[0], label.shape[2], label.shape[3])
        output = output.cpu().numpy().astype(int)
        label = label.cpu().numpy().astype(int)
        fwiou = evaluate(output, label)

        current_iter = epoch * len(train_loader) + i + 1
        current_lr = poly_learning_rate(args.base_lr, current_iter, max_iter, power=args.power)

        for index in range(0, args.index_split):
            optimizer.param_groups[index]['lr'] = current_lr / 10
        for index in range(args.index_split, len(optimizer.param_groups)):
            optimizer.param_groups[index]['lr'] = current_lr * 10

        if (i + 1) % args.print_freq == 0:
            print('Epoch: [{}/{}][{}/{}] '.format(epoch + 1, args.epochs, i + 1, len(train_loader)), "FWIOU: ", fwiou)


def validate_unet(val_loader, model, criterion):
    print('Start Evaluation')
    loss_sum = 0
    fwiou_sum = 0
    num_sum = 0

    model.eval()
    for i, (input, label) in enumerate(val_loader):
        input = input.cuda()
        label = label.cuda()
        output = model(input)
        label = label.view(label.shape[0], label.shape[2], label.shape[3])
        output = output.max(1)[1]
        output = output.cpu().numpy().astype(int)
        label = label.cpu().numpy().astype(int)
        fwiou = evaluate(output, label)

        fwiou_sum = fwiou_sum + fwiou * input.size(0)
        num_sum = num_sum + input.size(0)
        print(i*input.size(0))
    return fwiou_sum/num_sum

def main():
    fwiou_Best = 0
    parser = argparse.ArgumentParser(description="Remote Sensing Semantic Segmentation")
    parser.add_argument("--classes", default=8, help="num of class")
    parser.add_argument("--train_list", default="data/train1.csv", help="train dataset")
    parser.add_argument("--val_list", default="data/val1.csv", help="val dataset")
    parser.add_argument("--data_root", default="../../train/", help="data dir")
    parser.add_argument("--description", default="base_line")
    parser.add_argument("--arch", default="UNet")
    parser.add_argument("--train_gpu", default=[0,1,2,3])
    parser.add_argument("--workers", default=2, help="data loader workers")
    parser.add_argument("--batch_size", default=300)
    parser.add_argument("--batch_size_val", default=1)
    parser.add_argument("--base_lr", default=0.001)
    parser.add_argument("--epochs", default=340)
    parser.add_argument("--start_epoch", default=0)
    parser.add_argument("--index_split", default=1)
    parser.add_argument("--power", default=0.9)
    parser.add_argument("--momentum", default=0.9)
    parser.add_argument("--weight_decay", default=0.001)
    parser.add_argument("--print_freq", default=1)
    parser.add_argument("--save_freq", default=1)
    parser.add_argument("--save_path", default="results/UNet")
    parser.add_argument("--evaluate", default=True)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(str(x) for x in args.train_gpu)

    value_scale = 255
    mean = [0.355, 0.383, 0.359]
    mean = [item * value_scale for item in mean]
    std = [0.206, 0.202, 0.210]
    std = [item * value_scale for item in std]

    train_transform = transform.Compose([
        transform.RandRotate([-10, 10], padding=mean),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    val_transform = transform.Compose([
        transform.Normalize(mean=mean, std=std),
        transform.ToTensor()
    ])

    train_data = dataset.Mydataset(data_root=args.data_root, path=args.train_list,transform=train_transform)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=(train_sampler is None),
                                               num_workers=args.workers, pin_memory=True, sampler=train_sampler,
                                               drop_last=True)
    if args.evaluate:
        val_data = dataset.Mydataset(data_root=args.data_root, path=args.val_list, transform=val_transform)
        val_sampler = None
        val_loader = torch.utils.data.DataLoader(val_data, batch_size=args.batch_size_val, shuffle=False,
                                                 num_workers=args.workers, pin_memory=True, sampler=val_sampler)

    criterion = nn.CrossEntropyLoss().cuda()  #
    model = Unet("se_resnext50_32x4d", encoder_weights="imagenet", classes=8, criterion=criterion).cuda()

    modules_ori = [model.encoder]
    modules_new = [model.decoder]
    params_list = []
    for module in modules_ori:
       params_list.append(dict(params=module.parameters(), lr=args.base_lr / 10))
    for module in modules_new:
       params_list.append(dict(params=module.parameters(), lr=args.base_lr * 10))

    optimizer = torch.optim.Adam(params_list, lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=args.weight_decay, amsgrad=False)
    model = torch.nn.DataParallel(model).cuda()


    for epoch in range(args.start_epoch, args.epochs):
        epoch_now = epoch + 1
        train_unet(train_loader, model, optimizer, epoch, args)

        if args.evaluate:
            with torch.no_grad():
                val_Fwiou = validate_unet(val_loader, model, criterion)
            if (val_Fwiou > fwiou_Best):
                fwiou_Best = val_Fwiou
                if not os.path.exists(args.save_path):
                    os.mkdir(args.save_path)
                filename = os.path.join(args.save_path) + '/' + str(int(fwiou_Best)) + '.pth'
                torch.save({'epoch': epoch_now, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                           filename)
                print('Saved checkpoint')

if __name__ == '__main__':
    main()
