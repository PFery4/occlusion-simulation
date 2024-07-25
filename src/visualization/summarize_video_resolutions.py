import numpy as np
import os.path
import pandas as pd

import src.data.config as conf


if __name__ == '__main__':

    config = conf.get_config(conf.SDD_CONFIG)
    data_dir = os.path.join(config["path"], "annotations")
    print(data_dir)
    assert os.path.exists(data_dir)

    image_summaries_df = pd.DataFrame(
        columns=[
            'scene', 'video',
            'width [px]', 'height [px]', 'diagonal [px]',
            'width [m]', 'height [m]', 'diagonal [m]',
            'aspect ratio', 'm/px', 'px/m'
        ]
    )

    counter = 0
    for dir in os.scandir(data_dir):
        for video in os.scandir(dir):
            file = os.path.join(video, "reference.jpg")
            assert os.path.exists(file)
            counter += 1
            with open(file, "rb") as img_file:
                img_file.seek(163)

                a = img_file.read(2)
                height = (a[0] << 8) + a[1]
                a = img_file.read(2)
                width = (a[0] << 8) + a[1]

                # print(f"{os.path.basename(file)}: {width} x {height}")
                m_by_px = conf.COORD_CONV.loc[dir.name, video.name]['m/px']
                px_by_m = conf.COORD_CONV.loc[dir.name, video.name]['px/m']

                width_m = width * m_by_px
                height_m = height * m_by_px

                image_summaries_df.loc[len(image_summaries_df)] = [
                    dir.name, video.name,
                    width, height, np.sqrt(width ** 2 + height ** 2),
                    width_m, height_m, np.sqrt(width_m ** 2 + height_m ** 2),
                    np.min([width, height]) / np.max([width, height]), m_by_px, px_by_m
                ]

    S = 80
    image_summaries_df['min_padded_dim [m]'] = image_summaries_df['diagonal [m]'] + S
    image_summaries_df['min_pad_width [m]'] = (image_summaries_df['min_padded_dim [m]'] - image_summaries_df[
        'width [m]']) / 2
    image_summaries_df['min_pad_height [m]'] = (image_summaries_df['min_padded_dim [m]'] - image_summaries_df[
        'height [m]']) / 2
    image_summaries_df['min_pad_width [px]'] = image_summaries_df['min_pad_width [m]'] * image_summaries_df['px/m']
    image_summaries_df['min_pad_height [px]'] = image_summaries_df['min_pad_height [m]'] * image_summaries_df[
        'px/m']

    image_summaries_df.set_index(keys=['scene', 'video'], inplace=True)
    image_summaries_df.sort_index(inplace=True)
    print("Image resolution summary:")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 120):
        print(image_summaries_df)

    print()
    print(f"min width [px]: {min(image_summaries_df['width [px]'])}")
    print(f"max width [px]: {max(image_summaries_df['width [px]'])}")
    print(f"min height [px]: {min(image_summaries_df['height [px]'])}")
    print(f"max height [px]: {max(image_summaries_df['height [px]'])}")
    print(f"min px/m ratio: {min(conf.COORD_CONV['px/m'])}")
    print(f"max px/m ratio: {max(conf.COORD_CONV['px/m'])}")
    print(
        f"min pad value: {max(*image_summaries_df['min_pad_width [px]'], *image_summaries_df['min_pad_height [px]'])}")
