import pandas as pd
import os

models = [
    'unet_16_64_miou',
    'unet_16_128_miou',
    'unet_16_256_miou',
    'unet_32_128_miou',
    'unet_32_256_miou',
    'unet_32_512_miou',
    'unet_64_256_miou',
    'unet_64_512_miou',
    'unet_64_1024_miou',
    'unet_16_64_lovasz',
    'unet_16_128_lovasz',
    'unet_16_256_lovasz',
    'unet_32_128_lovasz',
    'unet_32_256_lovasz',
    'unet_32_512_lovasz',
    'unet_64_256_lovasz',
    'unet_64_512_lovasz',
    'unet_64_1024_lovasz',
    'bt_unet_16_64_miou',
    'bt_unet_16_128_miou',
    'bt_unet_16_256_miou',
    'bt_unet_32_128_miou',
    'bt_unet_32_256_miou',
    'bt_unet_32_512_miou',
    'bt_unet_64_256_miou',
    'bt_unet_64_512_miou',
    'bt_unet_64_1024_miou',
    'bt_unet_16_64_lovasz',
    'bt_unet_16_128_lovasz',
    'bt_unet_16_256_lovasz',
    'bt_unet_32_128_lovasz',
    'bt_unet_32_256_lovasz',
    'bt_unet_32_512_lovasz',
    'bt_unet_64_256_lovasz',
    'bt_unet_64_512_lovasz',
    'bt_unet_64_1024_lovasz',
    'unetpp_16_64_miou',
    'unetpp_16_128_miou',
    'unetpp_16_256_miou',
    'unetpp_32_128_miou',
    'unetpp_32_256_miou',
    'unetpp_32_512_miou',
    'unetpp_64_256_miou',
    'unetpp_64_512_miou',
    'unetpp_16_64_lovasz',
    'unetpp_16_128_lovasz',
    'unetpp_16_256_lovasz',
    'unetpp_32_128_lovasz',
    'unetpp_32_256_lovasz',
    'unetpp_32_512_lovasz',
    'unetpp_64_256_lovasz',
    'unetpp_64_512_lovasz'
]


def main():
    scores = {
        'acc': [],
        'avg_per_class_acc': [],
        'avg_jacc': [],
        'avg_f1': [],
        '0_acc': [],
        '1_acc': [],
        '2_acc': [],
        '3_acc': [],
        '4_acc': [],
        '0_jacc': [],
        '1_jacc': [],
        '2_jacc': [],
        '3_jacc': [],
        '4_jacc': [],
        '0_f1': [],
        '1_f1': [],
        '2_f1': [],
        '3_f1': [],
        '4_f1': [],
    }
    dirs = os.listdir('results')
    for model in models:
        for dir_ in dirs:
            if dir_.startswith(model):
                path = os.path.join('results', dir_, 'scores.csv')
                break
            else:
                path = ''
        if path == '':
            raise ValueError()
        # print(path)
        df = pd.read_csv(path)
        acc = df['overall_acc']
        avg_per_class_acc = df['avg_per_class_acc']
        avg_jacc = df['avg_jacc']
        avg_dice = df['avg_dice']
        pc_opa = df['pc_opa']
        pc_j = df['pc_j']
        pc_d = df['pc_d']
        scores['acc'].append(acc[0])
        scores['avg_per_class_acc'].append(avg_per_class_acc[0])
        scores['avg_jacc'].append(avg_jacc[0])
        scores['avg_f1'].append(avg_dice[0])
        scores['0_acc'].append(pc_opa[0])
        scores['1_acc'].append(pc_opa[1])
        scores['2_acc'].append(pc_opa[2])
        scores['3_acc'].append(pc_opa[3])
        scores['4_acc'].append(pc_opa[4])
        scores['0_jacc'].append(pc_j[0])
        scores['1_jacc'].append(pc_j[1])
        scores['2_jacc'].append(pc_j[2])
        scores['3_jacc'].append(pc_j[3])
        scores['4_jacc'].append(pc_j[4])
        scores['0_f1'].append(pc_d[0])
        scores['1_f1'].append(pc_d[1])
        scores['2_f1'].append(pc_d[2])
        scores['3_f1'].append(pc_d[3])
        scores['4_f1'].append(pc_d[4])

    df = pd.DataFrame(scores)
    df.to_csv(
        f'acc_scores.csv')


if __name__ == '__main__':
    main()
