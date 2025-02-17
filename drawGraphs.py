from lib.constants import *
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.pyplot import savefig
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sb
from queerbenchScore import QueerBenchScore
from lib.utils import getCSVFile  

FONT_SIZE_BIG = 14
FONT_SIZE = 11
NCOL = 2
PAULTOL_COLORBLINDPALETTE = ["#332288", "#88ccee", "#44aa99", "#117733", "#999933", "#ddcc77", "#cc6677", "#882255", "#aa4499", "#dddddd"]
HEATMAP_COLORBLINDPALETTE = ["#e4ff7a", "#ffe81a", "#ffbd00", "#ffa000", "#fc7f00"]

def afinnGraph(score_collection, model_list, subj_type):
    # Determine categories based on subject type
    categories = NOUN_TYPES if subj_type == NOUN else PRONOUN_CATEGORIES
    
    # Initialize graph data structure
    graph_data = defaultdict(lambda: defaultdict(list))
    df_data = defaultdict(list)
    
    # Prepare plot
    fig, ax = plt.subplots()
    ax.set_ylabel("Score", fontsize=FONT_SIZE_BIG)
    ax.set_xlabel("Model", fontsize=FONT_SIZE_BIG)
    plt.grid(linestyle='--', linewidth=0.5)
    
    # Adjust x-axis label rotation based on number of models
    rotation = 30 if len(model_list) > 8 else 20
    plt.xticks(rotation=rotation, rotation_mode="anchor", fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)
    #plt.setp(ax.xaxis.get_majorticklabels(), ha='right')

    # Populate graph data
    x_values = model_list
    for model in model_list:
        for category in categories:
            info_key = category if subj_type == NOUN else f"{PRONOUN} {category}"
            scores = score_collection[model][AFINN][info_key]
            for metric in [AVERAGE, STDEV, AFINN]:
                graph_data[category][metric].append(float(scores[metric]))

    # Plot each category with error bars
    for idx, category in enumerate(categories):
        ax.errorbar(
            x_values,
            graph_data[category][AVERAGE],
            yerr=graph_data[category][STDEV],
            linestyle=':',
            capsize=3,
            marker="o",
            transform=Affine2D().translate(float(f"0.{idx}"), 0.0) + ax.transData,
            label=category
        )
        df_data[category] = graph_data[category][AFINN]

    # Convert data to DataFrame and display
    df = pd.DataFrame.from_dict(json.loads(json.dumps(df_data)), orient='index', columns=model_list)
    print(df)

    # Finalize plot
    ax.legend(loc='best', ncol=NCOL)
    plt.tight_layout()
    os.makedirs(OUTPUT_GRAPHS, exist_ok=True)
    plt.savefig(f"{OUTPUT_GRAPHS}{subj_type}_{AFINN}.png", transparent=True)
    plt.show()

def hurtLexGraph(score_collection, model_list, subj_type):
    # Determine categories based on subject type
    categories = NOUN_TYPES if subj_type == NOUN else PRONOUN_CATEGORIES
    
    # Initialize graph data structure
    graph_data = defaultdict(lambda: defaultdict(list))
    line_data = defaultdict(list)
    
    # Prepare plot
    fig, ax = plt.subplots()
    ax.set_ylabel("Score", fontsize=FONT_SIZE_BIG)
    ax.set_xlabel("Model", fontsize=FONT_SIZE_BIG)
    plt.grid(linestyle='--', linewidth=0.5)
    #plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    
    # Adjust x-axis label rotation based on number of models
    rotation = 30 if len(model_list) > 8 else 20
    plt.xticks(rotation=rotation, rotation_mode="anchor", fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)

    # Populate graph data
    x_values = model_list
    for model in model_list:
        for category in categories:
            info_key = category if subj_type == NOUN else f"{PRONOUN} {category}"
            scores = score_collection[model][HURTLEX][info_key]
            for idx, metric in enumerate(set(HURTLEX_CATEGORIES_SHORT) - {TOTAL, HURTLEX}):
                percentage = ((int(scores[metric])/scores[TOTAL])*100) if metric in scores else 0
                graph_data[category][metric].append(percentage)
            line_data[category].append(scores[HURTLEX])

    # Generate x positions for the bars
    x = np.arange(len(x_values))
    bar_width = 0.75
    x_positions = [x - bar_width/3, x, x + bar_width/3]
    # Plot the bars and lines
    for jdx,category in enumerate(categories):
        for idx, metric in enumerate(set(HURTLEX_CATEGORIES_SHORT) - {TOTAL, HURTLEX}):
            bottom = np.zeros(len(x_values))
            if jdx == 0:
                ax.bar(x_positions[jdx], graph_data[category][metric], width=bar_width/3, label=f'{metric}',  color= PAULTOL_COLORBLINDPALETTE[idx], bottom=bottom, edgecolor = "white", linewidth = 0.5)
            else: 
                ax.bar(x_positions[jdx], graph_data[category][metric], width=bar_width/3, color= PAULTOL_COLORBLINDPALETTE[idx], bottom=bottom, edgecolor = "white", linewidth = 0.5)
            bottom += graph_data[category][metric]
    
    #ax.set_xticklabels(x_values)
    ax.legend(ncol = NCOL)
    plt.xticks(x, x_values)
    plt.legend()
    plt.show()

    
def drawGraph(score_collection, modelList):
    for subType in SUBJECT_TYPE:
        #afinnGraph(score_collection, modelList, subType)
        hurtLexGraph(score_collection, modelList, subType)
    
predictionsConsidered = 1
inputFolder = OUTPUT_EVALUATION
models = [ROBERTA_BASE, ROBERTA_LARGE]
score_collection = QueerBenchScore(inputFolder, models, predictionsConsidered)
drawGraph(score_collection, models)


