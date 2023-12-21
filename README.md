
## Running Tests

1. To run tests first create the dataset running the code in template_builder.ipynb using the following pattern.
Input: model, number of predictions

Example: 
```python
    TemplatePrediction(BERTTWEET_LARGE, 5)
```
2. To assess the dataset use the three files: 
- queer_evaluator - Hurtlex.ipynb
- queer_evaluator - Perspective.ipynb
- queer_evaluator - Sentiment Analysis.ipynb

In each file give as input: input file path, template, output file path

Example:
```python
    QueerEvaluator(EVALUATION_PATH, BERTWEET_LARGE_TEMPLATE_1, EVALUATION_PATH)
```
3. Run the functions in queer_results.ipynb to combine all the assessment, calculate the QueerBench score and generate the two overall scores for the terms and pronouns which are data/results/total_score_term.csv and data/results/total_score_pronouns.csv corrispondigly.

4. Use the functions in queer_graph.ipynb to generate the graphs based by the tables created in the previous step. The available functons are:
- Generate a point graphs giving model and subject categoies (used for AFFIN test). 
Example:
```python
error_bar(BERTWEET_MODELS, PRONOUN)
```
- Generate a line and bar graph (used for Perspective test). 
Example:
```python
perspective_linebar(perspective_linebar(ROBERTA_MODELS, PRONOUN))
```
- Generate a line and bar graph (used for Perspective test). 
Example:
```python
hurtlex_linebar(MODELS, TERM))
```
- Generate a two heat maps based on the pronouns and terms scores. The boolean in input allows to obtain a complete graph or a partial graph.
```python
term_graph_score(False)
pronouns_graph_score(False)
```

