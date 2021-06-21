import argparse
import pandas as pd
from tqdm.auto import tqdm
import time
# set up translation client
from google.cloud import translate_v2 as translate
import os


def main(args):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = args.google_application_credentials
    translate_client = translate.Client()
    df = pd.read_csv(args.input_fname, encoding='utf-8')
    
    src_list = df[args.source_lang].tolist()
    targ_list = df[args.target_lang].tolist()
    pred_list = []

    for i in tqdm(range(len(src_list) // args.batch_size)):
        start_idx = i * args.batch_size
        end_idx = (i + 1) * args.batch_size
        srcs = src_list[start_idx:end_idx]
        res = translate_client.translate(srcs, target_language=args.target_lang)
        time.sleep(1)
        pred_list += [i['translatedText'] for i in res]
        if i % (args.save_every / args.batch_size) == 0:
            save_df = pd.DataFrame({args.source_lang: src_list[:end_idx], 
                                    args.target_lang: targ_list[:end_idx], 
                                    f'pred_{args.target_lang}': pred_list[:end_idx]})
            save_df.to_csv(f'{args.input_fname[:-4]}_{end_idx}_gtranslated_{args.source_lang}_{args.target_lang}.csv', index=False)
    save_df = pd.DataFrame({args.source_lang: src_list[:end_idx], 
                            args.target_lang: targ_list[:end_idx], 
                            f'pred_{args.target_lang}': pred_list[:end_idx]})
    save_df.to_csv(f'{args.input_fname[:-4]}_{end_idx}_gtranslated_{args.source_lang}_{args.target_lang}.csv', index=False)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fname', type=str,
                        help='Input file to google translate')
    parser.add_argument('--source_lang', type=str,
                        help='Source langauge')
    parser.add_argument('--target_lang', type=str,
                        help='Target language')
    parser.add_argument('--batch_size', default=100,
                        type=int, help='Batch size to google translate')
    parser.add_argument('--save_every', default=50000,
                        type=int, help='Save every how many examples')
    parser.add_argument('--google_application_credentials', type=str,
                        default="/mnt/c/charin_projects/NLP-ZH_TH-Project/credentials/charicloud-45aef05e93db.json",
                        help='Credentials for Cloud Translation API')

    args = parser.parse_args()
    main(args)
