from constants import *

#KEY
PRONOUN = 'pronoun'
NOUN = 'noun'
SCORE = 'Score'
TOT = 'Tot'

NEO_ROW = 0
NEUTRAL_ROW = 1
BINARY_ROW = 2
QUEER_IDENTITY = 3
QUEER_ORIENTATION = 4
QUEER_OTHER = 5
NONQUEER_IDENTITY = 6
NONQUEER_ORIENTATION = 7
NONQUEER_OTHER = 8
QUEER_ROW = 9
NONQUEER_ROW = 10 

PERSPECTIVE_CATEGORIES = ['Toxicity', 'Identity attack', 'Insult', 'Profanity', 'Threat']
HURTLEX_CATEGORIES = ['Animals', 'Crime and immoral behavior', 'Derogatory words', 'Female genitalia', 'Male genitalia', 'Homosexuality', 'Potential negative connotations', 'Professions and occupations', 'Prostitution', 'Social and economic disadvantage']
HURTLEX_CATEGORIES_NAMES = ['AN', 'RE', 'CDS', 'ASF', 'ASM', 'OM', 'QAS', 'PA', 'PR', 'IS']
#IBM_COLORBLINDPALETTE = ["#648fff", "#785ef0", "#dc267f", "#fe6100", "#ffb000"]
PAULTOL_COLORBLINDPALETTE = ["#332288", "#88ccee", "#44aa99", "#117733", "#999933", "#ddcc77", "#cc6677", "#882255", "#aa4499", "#dddddd"]
HEATMAP_COLORBLINDPALETTE = ["#e4ff7a", "#ffe81a", "#ffbd00", "#ffa000", "#fc7f00"]
FONT_SIZE_BIG = 14
FONT_SIZE = 11
NCOL = 2

def afinnGraph(csv, type, models):
    x = []
    fig, ax = plt.subplots()
    plt.ylabel("Average score", fontsize=FONT_SIZE_BIG)
    plt.xlabel("Model", fontsize=FONT_SIZE_BIG)
    plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    if models == MODELS:
        plt.xticks(rotation=30, rotation_mode="anchor", fontsize = FONT_SIZE)
    else:
        plt.xticks(rotation=20, rotation_mode="anchor", fontsize = FONT_SIZE)
    plt.yticks(fontsize = FONT_SIZE)
    
    
    if type == PRONOUN:
        y1_neo, y2_neutral, y3_binary= [], [], []
        yerr1_neo, yerr2_neutral, yerr3_binary = [], [], []
        df_neo, df_neutral, df_binary = [], [], []
        for i in range(len(MODELS)):
            modelName = list(MODELS.keys())[i]
            x.append(modelName)
            y1_neo.append    (csv.loc[modelName, AFINN+ " (orig) "+ NEO])
            y2_neutral.append(csv.loc[modelName, AFINN+ " (orig) "+ NEUTRAL])
            y3_binary.append (csv.loc[modelName, AFINN+ " (orig) "+ BINARY])
            yerr1_neo.append(csv.loc[modelName]["StDev "+ NEO])
            yerr2_neutral.append(csv.loc[modelName]["StDev "+ NEUTRAL])
            yerr3_binary.append(csv.loc[modelName]["StDev "+ BINARY])
            df_neo.append    (csv.loc[modelName, AFINN+" "+ NEO])
            df_neutral.append(csv.loc[modelName, AFINN+" "+ NEUTRAL])
            df_binary.append (csv.loc[modelName, AFINN+" "+ BINARY])
            
        trans1 = Affine2D().translate(0.0, 0.0) + ax.transData
        trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
        trans3 = Affine2D().translate(+0.2, 0.0) + ax.transData
        er1 = ax.errorbar(x, y1_neo, yerr=yerr1_neo, linestyle = ':', capsize=3, marker="o", transform=trans1)
        er2 = ax.errorbar(x, y2_neutral, yerr=yerr2_neutral, linestyle = ':', capsize=3, marker="o", transform=trans2)
        er3 = ax.errorbar(x, y3_binary, yerr=yerr3_binary, linestyle = ':', capsize=3, marker="o", transform=trans3)
        ax.legend([NEO, NEUTRAL, BINARY], ncol = NCOL)
       
        df = pd.DataFrame(columns=models, index = ['Neo', 'Neutral', 'Binary'])
        df.loc['Neo'] = df_neo 
        df.loc['Neutral'] = df_neutral
        df.loc['Binary'] = df_binary
        print(df) 
    else:
        y1_queer, y2_non= [], []
        yerr1_queer, yerr2_non= [], []
        df_queer, df_non = [], []
        for i in range(len(MODELS)):
            modelName = list(MODELS.keys())[i]
            x.append(modelName)
            y1_queer.append(csv.loc[modelName, AFINN+ " (orig) "+QUEER])
            y2_non.append(csv.loc[modelName, AFINN+ " (orig) "+NONQUEER])
            yerr1_queer.append(csv.loc[modelName, "StDev "+ QUEER])
            yerr2_non.append(csv.loc[modelName, "StDev "+ NONQUEER])
            df_queer.append(f"\({csv.loc[modelName, AFINN+ ' (orig) '+QUEER]:.2f} \pm { csv.loc[modelName, 'StDev '+ QUEER]:.2f}\)")
            df_non.append(f"\({csv.loc[modelName, AFINN+ ' (orig) '+NONQUEER]:.2f} \pm {csv.loc[modelName, 'StDev '+ NONQUEER]:.2f}\)")
        
        trans1 = Affine2D().translate(0.0, 0.0) + ax.transData
        trans2 = Affine2D().translate(+0.1, 0.0) + ax.transData
        er1 = ax.errorbar(x, y1_queer, yerr=yerr1_queer, linestyle = ':', capsize=3, marker="o", transform=trans1)
        er2 = ax.errorbar(x, y2_non, yerr=yerr2_non, linestyle = ':', capsize=3, marker="o", transform=trans2)
        ax.legend([QUEER, NONQUEER], ncol = NCOL)

        df = pd.DataFrame(columns=models, index = ['Queer', 'Non Queer'])
        print(df)
        df.loc['Queer'] = df_queer 
        df.loc['Non Queer'] = df_non
        #display(df) 

    plt.grid(linestyle = '--', linewidth = 0.5)
    plt.tight_layout()
    os.makedirs(OUTPUT_GRAPHS, exist_ok=True)
    if models == MODELS:
        plt.savefig(OUTPUT_GRAPHS+ 'afinn_all.png', transparent=True)
    else: 
        plt.savefig(OUTPUT_GRAPHS+type+'/afinn_'+ (models[0].replace("base[1]", "")) +'.png', transparent=True)
    plt.show()
#afinn_graph(ALBERT_MODELS, noun)


def perspective_graph(models, type):
    fig, ax1 = plt.subplots()
    plt.setp(ax1.xaxis.get_majorticklabels(), ha='right')
    ax1.set_xlabel('Model', fontsize = FONT_SIZE_BIG)
    ax1.set_ylabel('Perspective score %', fontsize = FONT_SIZE_BIG)
    if models == MODELS:
        plt.xticks(rotation=30, rotation_mode="anchor", fontsize = FONT_SIZE)
    else:
        plt.xticks(rotation=20, rotation_mode="anchor", fontsize = FONT_SIZE)
    plt.yticks(fontsize = FONT_SIZE)
    plt.grid(linestyle = '--', linewidth = 0.5)

    if type == NOUN:
        bar_width = 0.40
        x_labels, bar_data1, bar_data2, line_data1, line_data2 = [], [], [], [], []
        tox_queer, id_queer, ins_queer, prof_queer, thre_queer, tox_non, id_non, ins_non, prof_non, thre_non = [], [], [], [], [], [], [], [], [], []
        data1 = [tox_queer, id_queer, ins_queer, prof_queer, thre_queer]
        data2 = [tox_non, id_non, ins_non, prof_non, thre_non]
        for m in models:
            csv = pd.read_csv(OUTPUT_GRAPHS+m+'_perspective.csv', sep=";", index_col=[0])
            x_labels.append(m)
            for ind, d in enumerate(data1):
                d.append((csv.loc[QUEER][PERSPECTIVE_CATEGORIES[ind]]/csv.loc[QUEER][TOT])*100)
            for ind, d in enumerate(data2):
                d.append((csv.loc[NONQUEER][PERSPECTIVE_CATEGORIES[ind]]/csv.loc[NONQUEER][TOT])*100)

            line_data1.append(csv.loc[QUEER][SCORE])
            line_data2.append(csv.loc[NONQUEER][SCORE])

        for d in data1:
            bar_data1.append(d)
        for d in data2:
            bar_data2.append(d)
        
        x = np.arange(len(x_labels))

        # Create the set of bars
        bottom1 = np.zeros(len(x_labels))
        for i, data in enumerate(bar_data1):
            ax1.bar(x - bar_width/2, data, width=bar_width, label=f'{PERSPECTIVE_CATEGORIES[i]}', color= PAULTOL_COLORBLINDPALETTE[i], bottom=bottom1,  edgecolor = "white", linewidth = 0.5)
            bottom1 += data

        bottom2 = np.zeros(len(x_labels))
        for i, data in enumerate(bar_data2):
            ax1.bar(x + bar_width/2, data, width=bar_width, color= PAULTOL_COLORBLINDPALETTE[i], bottom=bottom2,  edgecolor = "white", linewidth = 0.5)
            bottom2 += data

        ax1.plot(x - bar_width/2, line_data1, linestyle = 'None', marker = "o",  markersize=5, label=QUEER)
        ax1.plot(x + bar_width/2, line_data2, linestyle = 'None', marker = "^",  markersize=5, label=NONQUEER)
    else:
        bar_width = 0.75
        x_labels, bar_data1, bar_data2, bar_data3, line_data1, line_data2, line_data3 = [], [], [], [], [], [], []
        tox_neo, id_neo, ins_neo, prof_neo, thre_neo, tox_neu, id_neu, ins_neu, prof_neu, thre_neu, tox_bin, id_bin, ins_bin, prof_bin, thre_bin = [], [], [], [], [], [], [], [], [], [],[], [], [], [], []
        data1 = [tox_neo, id_neo, ins_neo, prof_neo, thre_neo]
        data2 = [tox_neu, id_neu, ins_neu, prof_neu, thre_neu]
        data3 = [tox_bin, id_bin, ins_bin, prof_bin, thre_bin]
        for m in models:
            csv = pd.read_csv(OUTPUT_GRAPHS+m+'_perspective.csv', sep=";", index_col=[0])
            x_labels.append(m)

            for ind, d in enumerate(data1):
                d.append((csv.loc[NEO][PERSPECTIVE_CATEGORIES[ind]]/csv.loc[NEO][TOT])*100)
            for ind, d in enumerate(data2):
                d.append((csv.loc[NEUTRAL][PERSPECTIVE_CATEGORIES[ind]]/csv.loc[NEUTRAL][TOT])*100)
            for ind, d in enumerate(data3):
                d.append((csv.loc[BINARY][PERSPECTIVE_CATEGORIES[ind]]/csv.loc[BINARY][TOT])*100)
                
            line_data1.append(csv.loc[NEO][SCORE])
            line_data2.append(csv.loc[NEUTRAL][SCORE])
            line_data3.append(csv.loc[BINARY][SCORE])

        for d in data1:
            bar_data1.append(d)
        for d in data2:
            bar_data2.append(d)
        for d in data3:
            bar_data3.append(d)
        
        # Create an array of x values for the bars
        x = np.arange(len(x_labels))
        
        # Create the first set of bars
        bottom1 = np.zeros(len(x_labels))
        for i, data in enumerate(bar_data1):
            ax1.bar(x - bar_width/3, data, width=bar_width/3, label=f'{PERSPECTIVE_CATEGORIES[i]}', color= PAULTOL_COLORBLINDPALETTE[i], bottom=bottom1,  edgecolor = "white", linewidth = 0.5)
            bottom1 += data

        bottom2 = np.zeros(len(x_labels))
        for i, data in enumerate(bar_data2):
            ax1.bar(x, data, width=bar_width/3, color= PAULTOL_COLORBLINDPALETTE[i], bottom=bottom2, edgecolor = "white", linewidth = 0.5)
            bottom2 += data

        bottom3 = np.zeros(len(x_labels))
        for i, data in enumerate(bar_data3):
            ax1.bar(x + bar_width/3, data, width=bar_width/3, color= PAULTOL_COLORBLINDPALETTE[i], bottom=bottom3, edgecolor = "white", linewidth = 0.5)
            bottom3 += data

        # Create the first set of line plots
        ax1.plot(x - bar_width/3, line_data1, linestyle = 'None', marker = "o",  markersize=5, label=NEO)
        ax1.plot(x , line_data2, linestyle = 'None', marker = "^",  markersize=5, label=NEUTRAL)
        ax1.plot(x + bar_width/3, line_data3, linestyle = 'None', marker = "s",  markersize=5, label=BINARY)

    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend(ncol = NCOL)
    plt.tight_layout()
    if models == MODELS:
        plt.savefig('../graphs/pronoun/'+ 'perspective_all.png', transparent=True)
    else: 
        plt.savefig('../graphs/'+type+'/perspective_'+ (models[0].replace("base[1]", "")) +'.png', transparent=True)
    plt.show()
    plt.close()


def hurtlex_graph(csv, type, models):
    
    fig, ax1 = plt.subplots()
    plt.setp(ax1.xaxis.get_majorticklabels(), ha='right')
    ax1.set_xlabel('Model', fontsize = FONT_SIZE_BIG)
    ax1.set_ylabel('Hurtlex score %', fontsize = FONT_SIZE_BIG)
    if models == MODELS:
        plt.xticks(rotation=30, rotation_mode="anchor", fontsize = FONT_SIZE)
    else:
        plt.xticks(rotation=20, rotation_mode="anchor", fontsize = FONT_SIZE)
    plt.yticks(fontsize = FONT_SIZE)
    plt.grid(linestyle = '--', linewidth = 0.5)

    if type == NOUN:
        bar_width = 0.40
        x_labels, bar_data1, bar_data2, line_data1, line_data2 = [], [], [], [], []
        an_queer, re_queer, cds_queer, asf_queer, asm_queer, om_queer, qas_queer, pa_queer, pr_queer, is_queer = [], [], [], [], [], [], [], [], [], []
        an_non, re_non, cds_non, asf_non, asm_non, om_non, qas_non, pa_non, pr_non, is_non = [], [], [], [], [], [], [], [], [], []
        data1 = [an_queer, re_queer, cds_queer, asf_queer, asm_queer, om_queer, qas_queer, pa_queer, pr_queer, is_queer]
        data2 = [an_non, re_non, cds_non, asf_non, asm_non, om_non, qas_non, pa_non, pr_non, is_non]
        for i in range(len(MODELS)):
            modelName = list(MODELS.keys())[i]
            x_labels.append(modelName)
            # for ind, d in enumerate(data1):
            #     d.append((csv.loc[QUEER][HURTLEX_CATEGORIES[ind]]/csv.loc[QUEER][TOT])*100)
            # for ind, d in enumerate(data2):
            #     d.append((csv.loc[NONQUEER][HURTLEX_CATEGORIES[ind]]/csv.loc[NONQUEER][TOT])*100)

            line_data1.append(csv.loc[modelName, HURTLEX + QUEER])
            line_data2.append(csv.loc[modelName, HURTLEX + NONQUEER])

        for d in data1:
            bar_data1.append(d)
        for d in data2:
            bar_data2.append(d)
        
        x = np.arange(len(x_labels))
        
        # Create the first set of bars
        bottom1 = np.zeros(len(x_labels))
        for i, data in enumerate(bar_data1):
            ax1.bar(x - bar_width/2, data, width=bar_width, label=f'{HURTLEX_CATEGORIES_NAMES[i]}',  color= PAULTOL_COLORBLINDPALETTE[i], bottom=bottom1, edgecolor = "white", linewidth = 0.5)
            bottom1 += data

        bottom2 = np.zeros(len(x_labels))
        for i, data in enumerate(bar_data2):
            ax1.bar(x + bar_width/2, data, width=bar_width, color= PAULTOL_COLORBLINDPALETTE[i], bottom=bottom2, edgecolor = "white", linewidth = 0.5)
            bottom2 += data

        # Create the first set of line plots
        ax1.plot(x - bar_width/2, line_data1, linestyle = 'None', marker = "o",  markersize=5, label=QUEER)
        ax1.plot(x + bar_width/2, line_data2, linestyle = 'None', marker = "^",  markersize=5, label=NONQUEER)
    else:
        bar_width = 0.75
        x_labels, bar_data1, bar_data2, bar_data3, line_data1, line_data2, line_data3 = [], [], [], [], [], [], []
        an_neo, re_neo, cds_neo, asf_neo, asm_neo, om_neo, qas_neo, pa_neo, pr_neo, is_neo = [], [], [], [], [], [], [], [], [], []
        an_neu, re_neu, cds_neu, asf_neu, asm_neu, om_neu, qas_neu, pa_neu, pr_neu, is_neu = [], [], [], [], [], [], [], [], [], []
        an_bin, re_bin, cds_bin, asf_bin, asm_bin, om_bin, qas_bin, pa_bin, pr_bin, is_bin = [], [], [], [], [], [], [], [], [], []
        data1 = [an_neo, re_neo, cds_neo, asf_neo, asm_neo, om_neo, qas_neo, pa_neo, pr_neo, is_neo]
        data2 = [an_neu, re_neu, cds_neu, asf_neu, asm_neu, om_neu, qas_neu, pa_neu, pr_neu, is_neu]
        data3 = [an_bin, re_bin, cds_bin, asf_bin, asm_bin, om_bin, qas_bin, pa_bin, pr_bin, is_bin]
        for modelName in models:
            csv = pd.read_csv(OUTPUT_GRAPHS+modelName+'_hurtlex.csv', sep=";", index_col=[0])
            x_labels.append(modelName)
            for ind, d in enumerate(data1):
                d.append((csv.loc[NEO][HURTLEX_CATEGORIES[ind]]/csv.loc[NEO][TOT])*100)
            for ind, d in enumerate(data2):
                d.append((csv.loc[NEUTRAL][HURTLEX_CATEGORIES[ind]]/csv.loc[NEUTRAL][TOT])*100)
            for ind, d in enumerate(data3):
                d.append((csv.loc[BINARY][HURTLEX_CATEGORIES[ind]]/csv.loc[BINARY][TOT])*100)

            line_data1.append(csv.loc[NEO][SCORE])
            line_data2.append(csv.loc[NEUTRAL][SCORE])
            line_data3.append(csv.loc[BINARY][SCORE])

        for d in data1:
            bar_data1.append(d)
        for d in data2:
            bar_data2.append(d)
        for d in data3:
            bar_data3.append(d)

        x = np.arange(len(x_labels))
        
        # Create the first set of bars
        bottom1 = np.zeros(len(x_labels))
        for i, data in enumerate(bar_data1):
            ax1.bar(x - bar_width/3, data, width=bar_width/3, label=f'{HURTLEX_CATEGORIES_NAMES[i]}',  color= PAULTOL_COLORBLINDPALETTE[i], bottom=bottom1, edgecolor = "white", linewidth = 0.5)
            bottom1 += data

        bottom2 = np.zeros(len(x_labels))
        for i, data in enumerate(bar_data2):
            ax1.bar(x, data, width=bar_width/3, color= PAULTOL_COLORBLINDPALETTE[i], bottom=bottom2, edgecolor = "white", linewidth = 0.5)
            bottom2 += data

        bottom3 = np.zeros(len(x_labels))
        for i, data in enumerate(bar_data3):
            ax1.bar(x + bar_width/3, data, width=bar_width/3, color= PAULTOL_COLORBLINDPALETTE[i], bottom=bottom3,  edgecolor = "white", linewidth = 0.5)
            bottom3 += data
        
        # Create the first set of line plots
        ax1.plot(x - bar_width/3, line_data1, linestyle = 'None', marker = "o",  markersize=5, label=NEO)
        ax1.plot(x , line_data2, linestyle = 'None', marker = "^",  markersize=5, label=NEUTRAL)
        ax1.plot(x + bar_width/3, line_data3, linestyle = 'None', marker = "s",  markersize=5, label=BINARY)

    
    ax1.set_xticks(x)
    ax1.set_xticklabels(x_labels)
    ax1.legend(ncol = NCOL)
    plt.tight_layout()
    if models == MODELS:
        plt.savefig('../graphs/pronoun/'+ 'hurtlex_all.png', transparent=True)
    else: 
        plt.savefig('../graphs/'+type+'/hurtlex_'+ (models[0].replace("base[1]", "")) +'.png', transparent=True)
    plt.show()
    plt.close()
#hurtlex_graph(BERT_MODELS, PRONOUN)


def partial_graph(type, test):
    if type == PRONOUN:
        data = pd.read_csv(OUTPUT_QUEERBENCH+'pronouns.csv', sep=";", index_col=[0]) 
        data = (data[:9] if test == True else data[9:])
    else:
        data = pd.read_csv(OUTPUT_QUEERBENCH+'nouns.csv', sep=";", index_col=[0]) 
        data = (data[:6] if test == True else data[6:])
    
    fig, xaxis = plt.subplots(figsize=(10, 6))
    dataplot = sb.heatmap(data, cmap=HEATMAP_COLORBLINDPALETTE, annot=True, linewidths=1) 
    plt.xticks(rotation=45, rotation_mode="anchor", fontsize = FONT_SIZE)
    plt.yticks(rotation=45, rotation_mode="anchor", fontsize = FONT_SIZE)
    plt.title("QueerBench score on "+("pronouns" if type == PRONOUN else "nouns"), fontsize = FONT_SIZE_BIG)
    plt.setp(dataplot.xaxis.get_majorticklabels(), ha='right')
    plt.setp(dataplot.yaxis.get_majorticklabels(), ha='right')
    plt.tight_layout()
    plt.savefig(OUTPUT_GRAPHS+type+ ('_test' if test is True else '_class')+ '.png', transparent=True)
    plt.show() 
    plt.close()
#partial_graph(NOUN, True)


#     testType = [PRONOUN, NOUN]
#     for t in testType:
#         partial_graph(t, True)
#         partial_graph(t, False)
#         for i in tqdm(range(len(MODELS))):
#             modelName = list(MODELS.keys())[i]
#             afinn_graph(modelName, t)
#             hurtlex_graph(modelName, t)
#             perspective_graph(modelName, t)
#     #total_table()
# saveAll()


def getTemplate(type, predictionsConsidered):
    os.makedirs(OUTPUT_QUEERBENCH, exist_ok=True)
    if os.path.exists(f"{OUTPUT_QUEERBENCH+type}_{predictionsConsidered}.csv"):
        try:
            return pd.read_csv(f"{OUTPUT_QUEERBENCH+type}_{predictionsConsidered}.csv", index_col=0)
        except:
            print("CSV file is broken")    
    else:
        print(f"There are no files related to the evaluation on {type} with {predictionsConsidered} prediction considered") 
        
#Input: input file path, template, output file path
predictionsConsidered = 1
testType = [ NOUN]

for t in testType:
    fileTemplate = getTemplate(t, predictionsConsidered)
#         partial_graph(t, True)
#         partial_graph(t, False)
    
    afinnGraph(fileTemplate, t, MODELS)
    hurtlex_graph(fileTemplate, t, MODELS)
    #perspective_graph(modelName, t)