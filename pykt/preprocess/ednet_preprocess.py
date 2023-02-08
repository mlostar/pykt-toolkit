import pandas as pd
import random
import os
from math import ceil
from .utils import sta_infos, write_txt


# Original SAINT uses part as the question concept
#KEYS = ["user_id", "tags", "question_id"]
KEYS = ["user_id","part","question_id"]
def read_data_from_csv(read_file, write_file):
    stares = []

    file_list = list()


    random.seed(2)
    samp = [i for i in range(840473)]
    random.shuffle(samp)

    count = 0

    for unum in samp:
        str_unum = str(unum)
        df_path = os.path.join(read_file, f"KT1/u{str_unum}.csv")
        if os.path.exists(df_path):
            df = pd.read_csv(df_path)
            df['user_id'] = unum

            file_list.append(df)
            count = count + 1

    # #     # if count == 5000:
    # #     #     break

    print(count)
    all_sa = pd.concat(file_list)
    all_sa["index"] = range(all_sa.shape[0])
    all_sa.to_csv(os.path.join(read_file, 'ednet_sample.csv'), index=False)
    # all_sa = pd.read_csv(os.path.join(read_file,'ednet_sample.csv'))
    print("Completed sampling")
    ca = pd.read_csv(os.path.join(read_file, 'contents', 'questions.csv'))
    ca['tags'] = ca['tags'].apply(lambda x:x.replace(";","_"))
    ca = ca[ca['tags']!='-1']
    co = all_sa.merge(ca, sort=False,how='left')
    # To avoid OOM when using the entire dataset
    del all_sa
    del ca
    
    #ins, us, qs, cs, avgins, avgcq, na = sta_infos(co, KEYS, stares)
    #print(f"original interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")

    co.dropna(inplace=True,subset=["user_id", "question_id", "elapsed_time", "timestamp", "tags", "user_answer"])
    co['correct'] = (co['correct_answer']==co['user_answer']).apply(int)
    # # To save memory
    co.drop(columns=['user_answer','correct_answer'],axis=1,inplace=True)
    
    #ins, us, qs, cs, avgins, avgcq, na = sta_infos(co, KEYS, stares)
    #print(f"after drop interaction num: {ins}, user num: {us}, question num: {qs}, concept num: {cs}, avg(ins) per s: {avgins}, avg(c) per q: {avgcq}, na: {na}")
    
    co.to_csv(os.path.join(read_file, 'ednet_sample_process.csv'), index=False)
    # Tags are removed since they are not used as concept key.
    # co = pd.read_csv(os.path.join(read_file,'ednet_sample_process.csv'),
        # usecols=['user_id','correct','question_id','timestamp','elapsed_time','index','part'],
        # dtype={'user_id':str,'correct':int,'question_id':str,'part':int,'elapsed_time':int})
    print("Completed sample processing")
    ui_df = co.groupby(['user_id'], sort=False)
    # To avoid OOM when using the entire dataset
    del co
    user_inters = []
    # Add chunk processing to avoid OOM
    chunk_counter = 0
    total_counter = 0
    chunk_item_counter = 0
    chunk_size = 5000
    user_count = len(ui_df)
    total_chunks = ceil(user_count / chunk_size)
    for ui in ui_df:
        user, tmp_inter = ui[0], ui[1]
        tmp_inter = tmp_inter.sort_values(by=["timestamp", "index"])
        seq_len = len(tmp_inter)
        # Use part as concept rather than tags
        #seq_skills = tmp_inter['tags'].astype(str)
        seq_skills = tmp_inter['part'].astype(str)
        seq_ans = tmp_inter['correct'].astype(str)
        seq_problems = tmp_inter['question_id'].astype(str)
        seq_start_time = tmp_inter['timestamp'].astype(str)
        seq_response_cost = tmp_inter['elapsed_time'].astype(str)

        assert seq_len == len(seq_problems) == len(seq_ans)

        user_inters.append(
            [[str(user), str(seq_len)], seq_problems, seq_skills, seq_ans, seq_start_time, seq_response_cost])
        
        chunk_item_counter+=1
        total_counter+=1
        if chunk_item_counter == chunk_size or total_counter == user_count:
            # Write in chunks
            write_txt(write_file, user_inters,chunks=True)
            chunk_item_counter=0
            user_inters = []
            chunk_counter+=1
            print(f"Completed writing chunk {chunk_counter}/{total_chunks}..")
    print("\n".join(stares))
    return


