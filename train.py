from torch.utils.data import DataLoader
import torch.nn as nn
import torch
from vgg16 import Vgg16, Test_model
from dataset import MyDataset
import time
import matplotlib.pyplot as plt

def chech_accuracy(loader, model):
    num_correct = 0
    num_samples = 0
    model.eval()
    # 不记录梯度
    with torch.no_grad():
        loss = 0
        for i, (x, y) in enumerate(loader):
            x = x.to(device=device)
            y = y.to(device=device)
            output = model(x)
            loss_c = loss_fc(outputs, labels)
            # 预测类别
            _, predict = output.max(1)
            # print(predict)
            # 预测正确的数量
            num_correct += (predict == y).sum()
            # 总样本数
            num_samples += predict.size(0)
            loss += loss_c.detach().cpu().numpy()
        loss = loss / len(loader)
    model.train()
    return num_correct.item(), num_samples, loss


if __name__ == '__main__':
    # 实例化模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = Vgg16(3).to(device)
    model = Test_model(3).to(device)
    # 交叉熵函数 多分类
    loss_fc = nn.CrossEntropyLoss()
    # 二分类
    # loss_fc = nn.BCEWithLogitsLoss()
    # 优化器
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    # 设置加载器
    train_data = MyDataset('img/train')
    train_loader = DataLoader(train_data, batch_size=12, shuffle=True)
    test_data = MyDataset('img/val')
    test_loader = DataLoader(test_data, batch_size=12, shuffle=True)

    # 训练模型
    best_test_acc = 0.1
    print("beginning")
    
    # Lists to store accuracy and loss values
    test_acc_values = []
    train_loss_values = []
    test_loss_values = []

    for epoch in range(1):
        start_time = time.time()
        loss = 0
        for i, (x, y) in enumerate(train_loader):
            images = x.to(device)
            labels = y.to(device)
            outputs = model(images)

            # 计算loss以及反向传播
            # loss_c = loss_fc(outputs.squeeze(dim=1), labels.float())
            loss_c = loss_fc(outputs, labels)
            loss_c.backward()
            opt.step()

            loss_c = loss_c.detach().cpu().numpy()
            loss += loss_c

        loss = loss / len(train_loader)

        num_correct, num_samples, test_loss = chech_accuracy(test_loader, model)
        test_acc = num_correct / num_samples

        # # Append accuracy and loss values to the lists
        test_acc_values.append(test_acc)
        train_loss_values.append(loss)
        test_loss_values.append(test_loss)

        print(type(test_acc))
        print(type(loss))
        print(type(test_loss))

        if best_test_acc < test_acc:
            best_test_acc = test_acc
            # 保存模型
            torch.save(model.state_dict(), 'save_model/best_model.pt')
            print(f'第{epoch}次，top_acc{test_acc}')
            print(f'epoch:{epoch}，best_acc:{best_test_acc},model saved successfully')
        print(
            f'epoch>>>{epoch} time {time.time() - start_time} train_loss>>>{loss:.4f}  test_acc>>>{test_acc:.4} test_loss>>>{test_loss}')

    # Plot accuracy and loss values
    plt.plot(test_acc_values, label='test accuracy')
    plt.plot(train_loss_values, label='train loss')
    plt.plot(test_loss_values, label='test loss')
    plt.xlabel('Epoch')
    plt.ylabel('Value')
    plt.legend()
    plt.show()
