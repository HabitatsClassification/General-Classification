# Habitat_Classification

## Brief Description

Machine Learning method to classify diferent habitats given predictor variables.


## How to execute the script

In order generate new processed data according new algorithms or changes in source code, you must delete files inside the folder data/computed.

It is possible to replicate generated data: :
    1. executing the script *executing Generalized_Classification_Function.R*; or,
    2. generate a website with details about the study from (*ux) command line: Rscript -e "rmarkdown::render_site(input = './website_source', encoding = 'UTF-8')"; or R prompt rmarkdown::render_site(input = './website_source', encoding = 'UTF-8').
    3. running the chunks of code using RStudio.
    
In the second case a complete website will be generated in the folder *docs*.
