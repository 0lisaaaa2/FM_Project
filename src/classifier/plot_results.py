import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


CLASSIFIER = 'classifier'
DATASET = 'dataset'
F1 = 'f1'
WIDTH = 0.35
BOTTOM_VALUE = 0.35
    
dataset_names = {
        'pneumonia': 'Chest',
        'bones_fixed': 'Bones', 
        'animals': 'Animals',
        'baggage': 'Baggage'
}
classifier_names = {
        "cxr": "CXR",
        "dinov3": "DINOv3"
}

if __name__ == "__main__":
    df = pd.read_csv('classifier_results_256.csv')
    test_df = df[df['split'] == 'test'].copy()

    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    datasets = ['pneumonia','bones_fixed', 'animals', 'baggage']
  

    display_names = [dataset_names[d] for d in datasets]
    
    
    classifiers = test_df[CLASSIFIER].unique()

    x = np.arange(len(datasets))
    
    classifier_colors = 	["#94b6ee", "#0057e7"]


    for i, classifier in enumerate(classifiers):
        f1_values = []
        for dataset in datasets:
            subset = test_df[(test_df[DATASET] == dataset) & 
                               (test_df[CLASSIFIER] == classifier)]
           
            f1_values.append(subset[F1].min())  # Use raw F1 values
            
        bar_heights = [f1 - BOTTOM_VALUE for f1 in f1_values]
        ax.bar(x + i*WIDTH, bar_heights, WIDTH, label=classifier_names[classifier], 
               color=classifier_colors[i], alpha=1, bottom=BOTTOM_VALUE)

    ax.set_yscale('logit')

    fontsize = 18
    ax.set_ylabel('F1 Score (Logit Scale)', fontsize=fontsize)
    ax.set_title('F1 Scores by Dataset and Classifier', fontsize=fontsize*1.2)
    ax.set_xticks(x + WIDTH/2)
    ax.set_xticklabels(display_names, fontsize=fontsize)
    ax.legend(fontsize=fontsize)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(axis='y', labelsize=fontsize)
    plt.tight_layout()
    plt.savefig('f1_bar_chart.png', dpi=300, bbox_inches='tight')