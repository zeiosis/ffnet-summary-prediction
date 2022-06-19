from re import findall, sub
import pandas as pd
from transformers import pipeline
import torch
    
droplist = ['Crime',
 'Fantasy',
 'General',
 'Horror',
 'Mystery',
 'Parody',
 'Poetry',
 'Sci_Fi',
 'Spiritual',
 'Supernatural',
 'Suspense',
 'Tragedy',
 'Western']

def webtext2dict(text):
    slistlite = findall(r"iews\n.*\nR", text)
    slist = []
    for x in slistlite:
        slist.append(x[5:len(x)-2])
        
    glistlite = findall(r"h - .*? - C|h - Ch", text)
    glist = []
    for x in glistlite:
        glist.append(x[4:len(x)-4])
    
    sumgenr_dict = {}
    for x in range(len(slist)):
        sumgenr_dict[slist[x]] = glist[x]
        
    keyremoveset = set()
    for key in sumgenr_dict:
        if len(sumgenr_dict[key]) > 30:
            keyremoveset.add(key)
    
    for x in keyremoveset:
        sumgenr_dict.pop(x, None)

    return sumgenr_dict

def dictclean(inputdict):
    #1st pass for Hurt/Comfort > Hurt_Comfort, s-f>s_f naming inconsistencies b/c vars dont support dashes
    #2nd pass to get rid of other slashes (k/j > [k, j])

    
    hc_repl_list = []
    
    for key in inputdict:
        if 'Hurt/Comfort' in inputdict[key]:
            hc_repl_list.append(key)
            
    for x in hc_repl_list:
        inputdict[x] = sub('Hurt/Comfort', 'Hurt_Comfort', inputdict[x])
        
        
    sf_repl_list = []
    
    for key in inputdict:
        if 'Sci-Fi' in inputdict[key]:
            sf_repl_list.append(key)
            
    for x in sf_repl_list:
        inputdict[x] = sub('Sci-Fi', 'Sci_Fi', inputdict[x])
    
    #1st pass done
    
    for key in inputdict:
        inputdict[key] = inputdict[key].split('/')
        
    #2nd pass done 
    
def ffdict_to_df(dfdict):
    df = pd.DataFrame()
    df = df.assign(Summary=0, Adventure=0, Angst=0, Crime=0, Drama=0, Family=0,\
                   Fantasy=0, Friendship=0, General=0, Horror=0, Humor=0,\
                   Hurt_Comfort=0, Mystery=0, Parody=0, Poetry=0, Romance=0,\
                   Sci_Fi=0, Spiritual=0, Supernatural=0, Suspense=0, Tragedy=0, Western=0) #works for ffnet.
    for x in dfdict:
        r2 = {}
        r2['Summary'] = x
        for values in dfdict[x]:
            if values != '':
                r2[values] = 1
        df = df.append(r2, ignore_index=True)
        
    df = df.fillna(0)
    return df

def ffcoll_to_df(filename, drop_rare=False):
    '''basically just go on ffnet and control a control c everything into a txt file.
    This works with multiple pages in one file as well.
    returns a DataFrame from a .txt filename'''
    f = open(filename, "r")
    rawtext1 = f.read()
    f.close 
    
    r1 = webtext2dict(rawtext1)
    dictclean(r1)
    print(r1)
    df = ffdict_to_df(r1)
    
    if drop_rare == False:
        return df
    else:
        df = df.drop(labels=droplist, axis=1)
        return df
    
def ffa(filename, tocsv_filename):
    '''Creates a csv containing summary and genre information, one-hotted, from a raw text file.'''
    
    from ff_analysis_tools import ffcoll_to_df
    
    df = ffcoll_to_df(filename)
    dfsums = df.sum()[1:]
    print(dfsums.sort_values())
    
    cutoff = int(input())
    
    dfsumdict = dict(df.sum()[1:])
    droplist = [x if dfsumdict[x] < cutoff else None for x in dfsumdict]
    droplist = list(filter(None, droplist))
    
    thintable = df.drop(labels=droplist, axis=1)
    
    k = []
    for x in range(len(df)): #can optimize pls this is so slow
        if x % 100 == 0:
            print(x / len(df) * 100)
        else:
            pass
        
        if sum(thintable.T[x][1:])==0:
            k.append(x)
        else:
            pass
        
    rowdrops = k
    cleanedtable = thintable.T.drop(labels=rowdrops, axis=1).T
    cleanedtable.to_csv(str(tocsv_filename) + '.csv')
    
    print(cleanedtable.sum()[1:].sort_values().plot.bar())

def prediction_error(df1, df2): #place one-hotted tables: truth vs predictions
    compdf = df1.compare(df2)
    single_wrong = 0
    double_wrong = 0
    for x in range(len(compdf)):
        if compdf.iloc[x].sum() == 2:
            double_wrong += 1 #probably the best metric imo, returns no. of 0-err, 1-err, 2-errs
        else:
            single_wrong += 1
    
    return {'0-wrong': len(df2) - len(compdf), '1-wrong': single_wrong, '2-wrong': double_wrong}
    
def accuracy_statistics(txt_src, csv_model_name_dict):
    '''Returns a barh with information about accuracy of predictions. 
    txt_src: the original scraped text file
    csv_model_name_dict: a dictionary containing information about the names of the csv prediction files and the names that will be shown in the chart. Should be formatted thus: {csv_predictions_file_1.csv: model_name_1, csv_predictions_file_2.csv: model_name_2 [...]}.'''
    
    by_new_test_sums_df = ffcoll_to_df(txt_src, drop_rare=True)
    
    N_summaries_df = by_new_test_sums_df.astype('int', errors='ignore')
    N_summaries_df = N_summaries_df.rename(columns={'Summary': 'summary'})
    
    N_df_names = {}
    for x in csv_model_name_dict:
        N_df_names[csv_model_name_dict[x]] = pd.read_csv(x)
        
    for x in N_df_names:
        print(x + ': ' + str(prediction_error(N_df_names[x], N_summaries_df)))
        
    N_reslist = []

    for x in N_df_names:
        k = []
        for y in prediction_error(N_df_names[x], N_summaries_df).values():
            k.append(y)
        N_reslist.append(k)
    
    N_resdf = pd.DataFrame(N_reslist, index=[x for x in N_df_names], columns=['0-wrong', '1-wrong', '2-wrong'])
    N_kresdf = N_resdf.applymap(lambda x: 100 * (x / (N_resdf.iloc[1].sum())))
    N_kresdf.plot.barh(stacked=True, color={'0-wrong':'green', '1-wrong':'yellow', '2-wrong':'red'})
    
def predict_labels(summary_list, model='zdreiosis/ff_analysis_2'): #defaults to 'zdreiosis/ff_analysis_2', might change in future
  #there should not be any reason to call this function more than once. If there is, rewrite, b/c it is v. slow
    classifier = pipeline(model=model, return_all_scores=True, device=cuda_id)
    return classifier(summary_list)

def preds_to_csv(predicted_labels, csvname):
    ksorted = [sorted(predicted_labels[y], key=lambda x: x['score']) for y in range(len(predicted_labels))]

    indexed_label_count = len(ksorted[0]) - 1
    reducedlist = [[x[indexed_label_count]['label'], x[indexed_label_count-1]['label']] for x in ksorted]

    strredlist = [','.join(x) for x in reducedlist]
    todf_predlist = [[summary_list[x], strredlist[x]] for x in range(len(strredlist))]

    preds_df = pd.DataFrame(todf_predlist, columns =['summary', 'tags'])
    preds_df = pd.concat([preds_df.drop('tags', axis=1), preds_df['tags'].str.get_dummies(',')], axis=1) #very useful code, ty stackoverflow

    preds_df.to_csv(csvname, index=False) 

def summaries_to_pred_csv(summary_list, model, csvname):
    '''use for zdr-types'''
    preds_to_csv(predict_labels(summary_list=summary_list, model=model), csvname)
    
def model_stats(modelname, test_list, csvname):
    '''use for other types'''
    classifier = pipeline(model=modelname, device=cuda_id)
    fsumlist = test_list

    fromsum = classifier(
      fsumlist,
      candidate_labels=['Adventure', 'Angst', 'Drama', 'Family', 'Friendship',
         'Humor', 'Hurt_Comfort', 'Romance']
    )

    strredlist = [str(fromsum[x]['labels'][0] + ',' + fromsum[x]['labels'][1]) for x in range(len(fromsum))]
    todf_predlist = [[summary_list[x], strredlist[x]] for x in range(len(strredlist))]

    preds_df = pd.DataFrame(todf_predlist, columns =['summary', 'tags'])
    preds_df = pd.concat([preds_df.drop('tags', axis=1), preds_df['tags'].str.get_dummies(',')], axis=1) #very useful code, ty stackoverflow
    preds_df.to_csv(str(str(csvname) + '.csv'), index=False) 
