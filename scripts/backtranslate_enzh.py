import argparse
import pandas as pd
from tqdm.auto import tqdm
#set up translation client
from google.cloud import translate_v2 as translate
import os



def main(args):
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"]=args.google_application_credentials
    translate_client = translate.Client()
    df = pd.read_csv(args.input_fname, encoding='utf-8')
    
    en_list = df.en.tolist()
    th_list = df.th.tolist()
    zh_list = []

    for i in tqdm(range(len(en_list)//args.batch_size)):
        start_idx = i*args.batch_size
        end_idx = (i+1)*args.batch_size
        ens = en_list[start_idx:end_idx]
        res = translate_client.translate(ens, target_language='zh_cn')
        zh_list+=[i['translatedText'] for i in res]
        if i % (1000/args.batch_size) == 0:
            save_df = pd.DataFrame({'en': en_list[:end_idx], 'th': th_list[:end_idx], 'zh_translated':zh_list[:end_idx]})
            save_df.to_csv(f'{args.input_fname[:-4]}_backtranslated.csv', index=False)
    save_df = pd.DataFrame({'en': en_list[:end_idx], 'th': th_list[:end_idx], 'zh_translated':zh_list[:end_idx]})
    save_df.to_csv(f'{args.input_fname[:-4]}_backtranslated.csv', index=False)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_fname', type=str, 
                       help='Input file to backtranslate')
    parser.add_argument('--batch_size', default=100,
                        type=int, help='Batch size to backtranslate')
    parser.add_argument('--google_application_credentials', type=str, 
                        default="/mnt/c/charin_projects/NLP-ZH_TH-Project/credentials/charicloud-45aef05e93db.json",
                       help='Credentials for Cloud Translation API')
    
    args = parser.parse_args()
    main(args)
    
    