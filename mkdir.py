import os

os.makedirs("GPU_LOG", exist_ok=True)
os.makedirs("results", exist_ok=True)
for ds in ['cifar', 'imagenet']:
    for bit in ['exact', 'qat', 'all8bit', 'only_weight', 'all4bit', 'forward8', 'forward4']:
        os.makedirs("results/{}/{}/model".format(ds, bit), exist_ok=True)

