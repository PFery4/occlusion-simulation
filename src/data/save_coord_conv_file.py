import os.path
import pandas as pd


if __name__ == '__main__':

    this_script_directory = os.path.abspath(os.path.dirname(__file__))
    config_dir = os.path.abspath(os.path.join(this_script_directory, '..', '..', 'config'))

    px_m_txt_file = os.path.join(config_dir, 'pixel_to_meter.txt')

    print(f"Extracting distance measurements data from:\n{px_m_txt_file}\n")
    px_per_m_df = pd.read_csv(
        px_m_txt_file, sep=', ', engine='python'
    )

    px_per_m_df['px/m'] = px_per_m_df['px'] / px_per_m_df['m']
    px_per_m_df['m/px'] = px_per_m_df['m'] / px_per_m_df['px']

    print("The coordinate conversion table is:")
    print(px_per_m_df)

    coord_conv_file_path = os.path.join(config_dir, 'coordinates_conversion.txt')
    print(f"\nSaving the coordinate conversion table as:\n{coord_conv_file_path}")
    px_per_m_df.to_csv(coord_conv_file_path, sep=';', index=False)
