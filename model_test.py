import torch
model = torch.load('./oct_resnet50_cosine.pth')
#print(model.keys())

new_state_dict = {}
i=0
for name, weight in model.items():
    if i < 6:
        new_state_dict[name] = weight
        i += 1
    else:
        ns = name.split('.')[0].split('r')
        if len(ns) == 2 and int(ns[1]) < 4:
            if '.0.' in name:
                new_state_dict[name] = weight
print(len(new_state_dict.keys()))
print(new_state_dict.keys())

torch.save(new_state_dict, './oct_resnet50_ours.pth')
