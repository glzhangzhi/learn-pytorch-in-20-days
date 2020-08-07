device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model.to(device)

features = features.to(device)
labels = labels.to(device)

model.compile(loss_func = nn.CrossEntropyLoss(),
             optimizer= torch.optim.Adam(model.parameters(),lr = 0.02),
             metrics_dict={"accuracy":accuracy},device = device) # 注意此处compile时指定了device

