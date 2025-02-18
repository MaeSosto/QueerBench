from lib.constants import *
import matplotlib.pyplot as plt
from matplotlib.transforms import Affine2D
from matplotlib.pyplot import savefig
from queerbenchScore import QueerBenchScore
import seaborn as sb 

FONT_SIZE_BIG = 14
FONT_SIZE = 11
NCOL = 2
PAULTOL_COLORBLINDPALETTE = ["#332288", "#88ccee", "#44aa99", "#117733", "#999933", "#ddcc77", "#cc6677", "#882255", "#aa4499", "#dddddd"]
HEATMAP_COLORBLINDPALETTE = ["#e4ff7a", "#ffe81a", "#ffbd00", "#ffa000", "#fc7f00"]

def lineGraph(score_collection, model_list, subj_type):
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
    plt.title(f"{AFINN} score on {subj_type}s", fontsize = FONT_SIZE_BIG)
    os.makedirs(OUTPUT_GRAPHS, exist_ok=True)
    plt.savefig(f"{OUTPUT_GRAPHS}{subj_type}_{AFINN}.png", transparent=True)
    #plt.show() 
    plt.close()

def barGraph(score_collection, model_list, subj_type, tool):
    # Determine categories based on subject type
    subjCategories = NOUN_TYPES if subj_type == NOUN else PRONOUN_CATEGORIES
    toolCategories = HURTLEX_CATEGORIES_SHORT if tool == HURTLEX else PERSPECTIVE_CATEGORIES
    
    # Initialize data structures
    graph_data = defaultdict(lambda: defaultdict(list))
    line_data = defaultdict(list)

    # Prepare plot
    fig, ax = plt.subplots()
    plt.setp(ax.xaxis.get_majorticklabels(), ha='right')
    ax.set_xlabel("Model", fontsize=FONT_SIZE_BIG)
    ax.set_ylabel(f"{tool} Score %", fontsize=FONT_SIZE_BIG)
    plt.grid(linestyle='--', linewidth=0.5)

    # Adjust x-axis label rotation
    rotation = 30 if len(model_list) > 8 else 20
    plt.xticks(rotation=rotation, rotation_mode="anchor", fontsize=FONT_SIZE)
    plt.yticks(fontsize=FONT_SIZE)

    # Populate graph data
    x_values = model_list
    for model in model_list:
        for category in subjCategories:
            info_key = category if subj_type == NOUN else f"{PRONOUN} {category}"
            scores = score_collection[model][tool][info_key]
            for metric in set(toolCategories) - {TOTAL, tool}:
                graph_data[category][metric].append((scores.get(metric, 0) * 100) / scores.get(TOTAL, 1))
            line_data[category].append(scores.get(tool, 0))

    # Generate x positions for the bars
    x = np.arange(len(x_values))
    bar_width = 0.75
    x_positions = [x - bar_width/3, x, x + bar_width/3]
    markers = ["o", "^", "s"]

    # Plot the bars and lines
    for jdx, category in enumerate(subjCategories):
        bottom = np.zeros(len(x_values))
        for idx, metric in enumerate(set(toolCategories) - {TOTAL, tool}):
            ax.bar(
                x_positions[jdx],
                graph_data[category][metric],
                width=bar_width/3,
                color=PAULTOL_COLORBLINDPALETTE[idx],
                bottom=bottom,
                edgecolor="white",
                linewidth=0.5,
                label= metric
            ) if jdx == 0 else ax.bar(
                x_positions[jdx],
                graph_data[category][metric],
                width=bar_width/3,
                color=PAULTOL_COLORBLINDPALETTE[idx],
                bottom=bottom,
                edgecolor="white",
                linewidth=0.5
            )
            bottom += graph_data[category][metric]
        # Plot the line for each category
        ax.plot(x_positions[jdx], line_data[category], linestyle='None', marker=markers[jdx], markersize=5, label=category)

    # Convert data to DataFrame and display
    df = pd.DataFrame.from_dict(json.loads(json.dumps(line_data)), orient='index', columns=model_list)
    print(df)
    
    # Finalize plot
    ax.legend(ncol=NCOL)
    plt.xticks(x, x_values)
    plt.title(f"{tool} score on {subj_type}s", fontsize = FONT_SIZE_BIG)
    plt.savefig(f"{OUTPUT_GRAPHS}{subj_type}_{tool}.png", transparent=True)
    #plt.show() 
    plt.close()

def heatMap(subj_type, filter = ""):
    data = pd.read_csv(OUTPUT_QUEERBENCH + subj_type + '.csv', index_col=[0]) 
    df_filtered = data[~data.index.str.contains(QUEERBENCH)] if filter == "" else data[data.index.str.contains(QUEERBENCH)]
    
    fig, xaxis = plt.subplots(figsize=(10, 6))
    dataplot = sb.heatmap(df_filtered, cmap=HEATMAP_COLORBLINDPALETTE, annot=True, linewidths=1) 
 
    plt.xticks(rotation=45, rotation_mode="anchor", fontsize = FONT_SIZE)
    plt.yticks(rotation=45, rotation_mode="anchor", fontsize = FONT_SIZE)
    plt.title(f"QueerBench score on {subj_type}s", fontsize = FONT_SIZE_BIG)
    plt.setp(dataplot.xaxis.get_majorticklabels(), ha='right')
    plt.setp(dataplot.yaxis.get_majorticklabels(), ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_GRAPHS}{filter+"_" if filter != "" else ""}{subj_type}.png', transparent=True)
    ##plt.show()   
    plt.close() 
    
def drawGraph(score_collection, modelList):
    for subType in SUBJECT_TYPE:
        lineGraph(score_collection, modelList, subType)
        barGraph(score_collection, modelList, subType, HURTLEX)
        barGraph(score_collection, modelList, subType, PERSPECTIVE)    
        heatMap(subType)
        heatMap(subType, QUEERBENCH)
        
predictionsConsidered = 1
inputFolder = OUTPUT_EVALUATION
models = MODEL_LIST
score_collection = QueerBenchScore(inputFolder, models, predictionsConsidered)
drawGraph(score_collection, models)


