# ffnet-summary-prediction
NLP model for predicting FFnet genre tags from summary.

dangerzone/ contains files only of historical value, used during development; they are not meant to be used further. 
currently working on making the whole thing actually understandable/usable: progress 90%...

<h3>ff_analysis_tools</h3>
use class FFWebtext to reference raw web-scraped text, whether in string form or as a reference to a .txt file. Each work is represented by a Summary object, which initially contains information about the summary and genres of the work; the predicted genres (pred_genre1/2) can be modified by a SummarySet object, which is used to access HF models to use to predict genre values.

<h3>Example Summary object:</h3>

    s1 = Summary('She had a tiny, tiny, tiny crush on him. Maybe.', 'Romance', 'Drama', 'Romance', 'Hurt_Comfort')
    
    s1.pred_acc()
    >>> '2t1r'

<h3>Example FFWebtext object:</h3>

    ffwt1 = FFWebtext(filename='ffdump7.txt')

    ffwt1.to_csv('ffdump7.csv)

    pd.read_csv('ffdump7.csv)
    >>> <DataFrame>


