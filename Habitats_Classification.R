library(caretEnsemble)
library(caret)
library(dplyr)
library(here)
library(pROC)
library(doParallel)
cl <- makePSOCKcluster(3)
registerDoParallel(cl)
#### Create/Open Data ####
#algo <- c("kknn","rf", "LogitBoost", "nnet", "svmRadial")
#algo <- c("RRF", "ranger", #"ORFridge", # Random Forest 
#          "LogitBoost","vglmContRatio", "vglmCumulative", # Logistic Regression
#          "svmRadialCost", "svmRadialSigma", "svmRadial", #SVM
#          "pcaNNet", "avNNet", "rbfDDA") # NeuralNetwork
#
algo <- c(# NeuralNetwork
              "avNNet", "dnn", "mlp", "mlpML", "mlpWeightDecay",
              "mlpWeightDecayML", "monmlp", "multinom", "nnet",
              "pcaNNet", "rbfDDA", 
          # Random Forest
              "cforest", "extraTrees", "parRF", "ranger", "Rborist",
              "rf", "rFerns", "RRF", "RRFglobal", "wsrf",                
          # Logistic Regression
              "LMT", "LogitBoost", "polr", "vglmAdjCat", "vglmContRatio", "vglmCumulative",
          # SVM
              "lssvmRadial", "svmLinear", "svmLinear2", "svmRadialWeights",
              "svmRadialSigma", "svmRadialCost", "svmRadial", "svmPoly")     

#tag <- read.csv("tag_data.csv", row.names = 1)
#tag <- as.matrix(tag)
#algo2 <- tag[tag[,"Classification"] == 1 & 
#             tag[,"Two.Class.Only"] == 0 & 
#             tag[,"Categorical.Predictors.Only"] == 0 & 
#             tag[,"Binary.Predictors.Only"] == 0 &
#             tag[,"Random.Forest"] == 0 | 
#             tag[,"Logistic.Regression"] == 0 | 
#             tag[,"Neural.Network"] == 0 | 
#             tag[,"Support.Vector.Machines"] == 0 ,]
#algo <- rownames(algo)
#algo <- gsub("(?<=\\()[^()]*(?=\\))(*SKIP)(*F)|.", "", algo, perl=T)
#algo <- algo[-51]
#algo <- algo[-c(6, 8, 10, 22, 27, 28)]
#
##elmNN, gpls, logicFS, FCNN4R, mxnet
##   elm[6],gpls[8],logicBag[10]  ,mlpSGD[22], mxnet[27], mxnetAdam[28]
#th <- 0.6
#result <- multiclass(train, test, algo[1:14])
#result2 <- multiclass(train, test, algo[15:28])
#result3 <- multiclass(train, test, algo[29:42])
#result4 <- multiclass(train, test, algo[43:56])
#
#

df <- read.csv(here("NTT_xgboost.csv"))
df <- df[,-1]
levels(df$vegetation.type) <- gsub("Coastal white-sand woodland"               ,"Restinga",       levels(df$vegetation.type))
levels(df$vegetation.type) <- gsub("Evergreen cloud dwarf-forest"              ,"Rainforest",     levels(df$vegetation.type))
levels(df$vegetation.type) <- gsub("Evergreen cloud forest"                    ,"Rainforest",     levels(df$vegetation.type))
levels(df$vegetation.type) <- gsub("Evergreen Rain forest"                     ,"Rainforest",     levels(df$vegetation.type))
levels(df$vegetation.type) <- gsub("Mixed Araucaria forest"                    ,"High.Elevation", levels(df$vegetation.type))
levels(df$vegetation.type) <- gsub("Rocky highland seasonal dwarf-forest"      ,"Rocky",          levels(df$vegetation.type))
levels(df$vegetation.type) <- gsub("Rocky highland seasonal savanna"           ,"Rocky",          levels(df$vegetation.type))
levels(df$vegetation.type) <- gsub("Seasonal semideciduous forest"             ,"Semideciduous",  levels(df$vegetation.type))
levels(df$vegetation.type) <- gsub("Subtropical riverine semideciduous forest" ,"High.Elevation", levels(df$vegetation.type))
levels(df$vegetation.type) <- gsub("Tropical riverine semideciduous forest"    ,"Riverine",       levels(df$vegetation.type))
head(df)

classification <- function(df, th){
  habitats <- levels(df[,1])
  nr <- createDataPartition(df[,1], p=0.9, list=FALSE)
  train <- df[nr,]
  test <- df[-nr,]
  #### Generatind Methods ####
  multiclass <- function(train, test, algo, th){
    t <- trainControl(
        method = "repeatedcv", 
        number = 10, 
        index = createFolds(train[,1], 10),
        repeats = 10, 
        savePredictions = 'final', 
        classProbs = TRUE, 
        summaryFunction = multiClassSummary,
        allowParallel = F, 
        sampling = "up", 
        returnResamp = "final",
        selectionFunction = "best"
      )
    ensemble <- caretList(
        reformulate(termlabels = colnames(train)[-1], response=colnames(train)[1]),
        data=train, 
        trControl=t, 
        methodList=algo[1:2], 
        continue_on_fail=T,
        metric="AUC"
      )
    model_preds <- lapply(ensemble, predict, newdata=test, type="prob")
    
    #oneSE(ensemble[[1]]$results, "AUC", num = 10, maximize = T)
    values_auc <- NA
    for (i in 1:length(algo)) {
      v <- ensemble[[i]]$results$AUC[best(ensemble[[i]]$results, "AUC", maximize = T)]
      if(v > th){
        assign(paste0("auc_",algo[i]), v)
        values_auc[i] <- v
      } else {
        assign(paste0("auc_",algo[i]), 0)
        values_auc[i] <- 0
      }
    }
    sum_auc <- sum(values_auc)
    result <- test
    for (i in 1:length(habitats)){
      df2 <- sapply(model_preds,'[',i)
      df2 <- Map('*',df2,values_auc) # multiplica previsões pelo AUC
      df2 <- as.data.frame(df2) 
      df2 <- rowSums(df2)/sum_auc 
      result <- cbind(result,df2)
    }
    colnames(result) <- c(colnames(test), habitats)
    final_prediction <- apply(result[,-(1:ncol(df))],1,function(x) names(which.max(x)))
    habitat_probs <- NA
    for(i in 1:nrow(result)){
      habitat_probs[i] <- result[i,colnames(result) == result[,1][i]]
    }
    result <- cbind(result, final_prediction, habitat_probs)
    
    return(result)
  }
  one_vs_all <- function(train, test, algo, th){
    
    for (i in 1:length(habitats)){
     train2 <- train
     train2[,1] <- ifelse(train2[,1] != habitats[i], "Other", habitats[i]) %>% 
     factor(levels = c(habitats[i], "Other"))
     assign(paste0(habitats[i]),train2)
     print(i)
    }

    habitat_classification <- function(df_habitat, algo, th){
      u <- unique(df_habitat[,1])
      u <- u[u != "Other"]
      t <- trainControl(method = "repeatedcv", 
                        number = 10, 
                        index = createFolds(df_habitat[,1], 10),
                        repeats = 10, 
                        savePredictions = 'final', 
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary,
                        allowParallel = F, 
                        sampling = "down")
      ensemble <-  caretList(reformulate(termlabels = colnames(train)[-1], response=colnames(train)[1]), 
                             data=df_habitat, 
                             trControl=t, 
                             methodList=algo, 
                             continue_on_fail=T,
                             metric="ROC",
                             verbose=T)
      assign(paste0(u,"_",algo[1]), ensemble)
    }
    
    # Run Function:
    Rainforest_result     <- habitat_classification(Rainforest     , algo, th)
    Riverine_result       <- habitat_classification(Riverine       , algo, th)
    Rocky_result          <- habitat_classification(Rocky          , algo, th)
    Restinga_result       <- habitat_classification(Restinga       , algo, th)
    Semideciduous_result  <- habitat_classification(Semideciduous  , algo, th)
    High.Elevation_result <- habitat_classification(High.Elevation , algo, th)
    
    ensembles <- paste0(habitats,"_result")

    result <- test
    
    for (j in 1:length(ensembles)){
      print(j)
      en <- get(ensembles[j])
      
      values_roc <- NA
      for (i in 1:length(algo)) {
        v <- en[[i]]$results$ROC[best(en[[i]]$results, "ROC", maximize = T)]
        if(v > th){
          assign(paste0("roc_",algo[i]), v)
          values_roc[i] <- v
        } else {
          assign(paste0("roc_",algo[i]), 0)
          values_roc[i] <- 0
        }
      }
    
      sum_roc <- sum(values_roc)
      model_preds <- lapply(en, predict, newdata=test, type="prob")
      df2 <- sapply(model_preds,'[',habitats[j])
      df2 <- Map('*',df2,values_roc)
      df2 <- as.data.frame(df2)
      df2 <- rowSums(df2)/sum_roc
      result <- cbind(result,df2)
    }
    
    colnames(result) <- c(colnames(test), habitats)
    
    final_prediction <- colnames(result[,-(1:4)])[apply(result[,-(1:4)],1,which.max)]
    habitat_probs <- NA
    for(i in 1:nrow(result)){
      habitat_probs[i] <- result[i,colnames(result) == result[,1][i]]
    }
    result <- cbind(result, final_prediction, habitat_probs)
    return(result)
  }
  result_multiclass <- multiclass(train, test, algo[1:2], th)
  result_one_vs_all <- one_vs_all(train, test, algo, th)
  #### Comparing Methods ####
  mc.roc <- multiclass.roc(result_multiclass[,1], result_multiclass$habitat_probs)
  oa.roc <- multiclass.roc(result_one_vs_all[,1], result_one_vs_all$habitat_probs)
  result <- list("Multiclass" = mc.roc, "One.vs.all" = oa.roc)
  return(result)
}

n <- 1
result_multiclass_100_newalgo <- replicate(n, classification(df, 0.6))

even <- seq(2,200,2)
odd <- seq(1,by=2, len=100)

auc_multiclass_100 <- result_multiclass_100[odd]



y <- NA
for (i in 1:100){
  y <- c(x, result_multiclass_100[,i]$One.vs.all$auc)
  #x <- result_multiclass_100[,i]
  #multiclass.roc(x$vegetation.type, x$roc_probs, direction = "auto")
  #auc_multiclass_100[i] <- x2$auc
}
y <- y[-1]

# Other Materials:
#### 1. Multi-class Classification #### 
#multiclass <- function(train, test, algo){
#  t <- trainControl(method = "repeatedcv", number = 10, index = createFolds(train$vegetation.type, 10),
#                    repeats = 10, savePredictions = 'final', classProbs = TRUE, summaryFunction = multiClassSummary,
#                    allowParallel = F, sampling = "up", returnResamp = "final")
#  ensemble <- caretList(vegetation.type ~ ., data=train, trControl=t, methodList=algo, continue_on_fail=T)
#  model_preds <- lapply(ensemble, predict, newdata=test, type="prob")
#  vote <- function(x){colnames(x)[max.col(x,ties.method="first")]}
#  result <- test
#  result2 <- test
#  for (i in 1:length(algo)){
#    m <- model_preds[[i]]
#    result <- cbind(result, vote(m))
#    colnames(result)[4+i] <- paste0(algo[i])
#    result2 <- cbind(result2, apply(m, 1, max))
#    colnames(result2)[4+i] <- paste0(algo[i])
#    print(i)
#  }
#  # Elaborar soma de votos simples:
#  votes <- NA
#  simple.sum <- apply(result[,-(1:4)],1,function(x) names(which.max(table(x))))
#  for(i in 1:length(simple.sum)){
#    votes[i] <- rowSums(result[i,-(1:4)] == simple.sum[i])
#  }
#  result <- cbind(result, simple.sum, votes)
#  return(result)
#}
#### 2. One-vs-all Classification #### 
#one_vs_all <- function(train, test, algo){
#  Rainforest <- train
#  Rainforest$vegetation.type <- ifelse(Rainforest$vegetation.type != "Rainforest", "Other", "Rainforest") %>% 
#                                factor(levels = c("Rainforest", "Other"))
#  High.Elevation <- train
#  High.Elevation$vegetation.type <- ifelse(High.Elevation$vegetation.type != "High.Elevation", "Other", "High.Elevation") %>% 
#                                    factor(levels = c("High.Elevation", "Other"))
#  Semideciduous <- train
#  Semideciduous$vegetation.type <- ifelse(Semideciduous$vegetation.type != "Semideciduous", "Other", "Semideciduous") %>% 
#                                    factor(levels = c("Semideciduous", "Other"))
#  Restinga <- train
#  Restinga$vegetation.type <- ifelse(Restinga$vegetation.type != "Restinga", "Other", "Restinga") %>% 
#                                    factor(levels = c("Restinga", "Other"))
#  Riverine <- train
#  Riverine$vegetation.type <- ifelse(Riverine$vegetation.type != "Riverine", "Other", "Riverine") %>% 
#                                    factor(levels = c("Riverine", "Other"))
#  Rocky <- train
#  Rocky$vegetation.type <- ifelse(Rocky$vegetation.type != "Rocky", "Other", "Rocky") %>% 
#                                    factor(levels = c("Rocky", "Other"))
#  
#  habitat_classification <- function(df, algo){
#    nr <- createDataPartition(df$vegetation.type, p=0.9, list=FALSE)
#    train2 <- df[nr,]
#    u <- unique(df$vegetation.type)
#    u <- u[u != "Other"]
#    t <- trainControl(method = "repeatedcv", number = 10, index = createFolds(train2$vegetation.type, 10),
#                      repeats = 10, savePredictions = 'final', classProbs = TRUE, summaryFunction = twoClassSummary,
#                      allowParallel = F, sampling = "down")
#    ensemble <-  caretList(vegetation.type ~ ., data=train2, trControl=t, methodList=algo, continue_on_fail=T,  verbose=T)
#    assign(paste0(u,"_",algo[1]), ensemble)
#  }
#  
#  # Run Function:
#  result_Rainforest     <- habitat_classification(Rainforest     , algo)
#  result_Riverine       <- habitat_classification(Riverine       , algo)
#  result_Rocky          <- habitat_classification(Rocky          , algo)
#  result_Restinga       <- habitat_classification(Restinga       , algo)
#  result_Semideciduous  <- habitat_classification(Semideciduous  , algo)
#  result_High.Elevation <- habitat_classification(High.Elevation , algo)
#  
#  #e <- extractPrediction(result_Rainforest, testX = test[,2:4], testY = test[,1])
#  #e <- e[e$dataType == "Test",]
#  
#  result <- test
#  result2 <- test
#  result3 <- test
#  ensembles <- c("result_Rainforest",
#                 "result_Riverine",
#                 "result_Rocky",
#                 "result_Restinga",
#                 "result_Semideciduous",
#                 "result_High.Elevation")
#  
#  vote <- function(x){colnames(x)[max.col(x,ties.method="first")]}
#  
#  for (j in 1:length(ensembles)){
#    print(1)
#    model_preds <- lapply(get(ensembles[j]), predict, newdata=test, type="prob") # probs 
#    print(2)
#    for (i in 1:length(algo)){
#      m <- model_preds[[i]]
#      
#      votos <- vote(m)
#      result[,paste0(ensembles[j],"_",algo[i])] <- votos
#      
#      nvotos <- m[, colnames(m) != "Other"]
#      result2[,paste0(ensembles[j],"_",algo[i])] <- nvotos
#      
#      print(i)
#    }
#    print(ncol(result2))
#    result3[,paste0(ensembles[j])] <- rowMeans(result2[,(ncol(result2)-4):ncol(result2)])
#  }
#  
#  head(result2)
#  
#  
#  final_prediction <- colnames(result3[,-(1:4)])[apply(result3[,-(1:4)],1,which.max)]
#  final_prediction <- substr(final_prediction, 8, nchar(final_prediction))
#  final_prediction <- cbind(test, final_prediction)
#  final_prediction[,"probability"] <- apply(result3[,-(1:4)],1,max)
#  return(final_prediction)
#}
#
#
## Algoritmos:
#algo <- c("avNNet"              ,"bayesglm"            ,"brnn"               , "cforest"            ,
#          "dnn"                 ,"extraTrees"          ,"LMT"                , "LogitBoost"         ,
#          "logreg"              ,"lssvmLinear"         ,"lssvmPoly"          , "lssvmRadial"        ,
#          "mlp"                 ,"mlpKerasDecay"       ,"mlpKerasDecayCost"  , "mlpKerasDropout"    ,
#          "mlpKerasDropoutCost" ,"mlpML"               ,"mlpWeightDecay"     , "mlpWeightDecayML"   ,
#          "monmlp"              ,"multinom"            ,"neuralnet"          , "nnet"               ,
#          "ordinalRF"           ,"ORFlog"              ,"parRF"              , "pcaNNet"            ,
#          "plr"                 ,"polr"                ,"qrnn"               , "ranger"             ,
#          "rbf"                 ,"rbfDDA"              ,"Rborist"            , "rf"                 ,
#          "rFerns"              ,"rfRules"             ,"RRF"                , "RRFglobal"          ,
#          "svmBoundrangeString" ,"svmExpoString"       ,"svmLinear"          , "svmLinear2"         ,
#          "svmLinearWeights"    ,"svmLinearWeights2"   ,"svmPoly"            , "svmRadial"          ,
#          "svmRadialCost"       ,"svmRadialSigma"      ,"svmRadialWeights"   , "svmSpectrumString"  ,
#          "vglmAdjCat"          ,"vglmContRatio"       ,"vglmCumulative"     , "wsrf"     )
#
## Algoritmos funcionando: #33 trava tudo, svmSpectrumString precisa de matrix
#algo_funcionando <- c("avNNet",              # NeuralNetwork
#                      "dnn",                 # NeuralNetwork
#                      "mlp",                 # NeuralNetwork
#                      "mlpML",               # NeuralNetwork
#                      "mlpWeightDecay",      # NeuralNetwork
#                      "mlpWeightDecayML",    # NeuralNetwork
#                      "monmlp",              # NeuralNetwork
#                      "multinom",            # NeuralNetwork
#                      "nnet",                # NeuralNetwork
#                      "pcaNNet",             # NeuralNetwork
#                      "rbfDDA",              # NeuralNetwork
#                      
#                      "cforest",             # Random Forest
#                      "extraTrees",          # Random Forest
#                      "parRF",               # Random Forest
#                      "ranger",              # Random Forest
#                      "Rborist",             # Random Forest
#                      "rf",                  # Random Forest
#                      "rFerns",              # Random Forest
#                      "RRF",                 # Random Forest
#                      "RRFglobal",           # Random Forest
#                      "wsrf",                # Random Forest
#                      
#                      "LMT",                 # Logistic Regression
#                      "LogitBoost",          # Logistic Regression
#                      "polr" ,               # Logistic Regression
#                      "vglmAdjCat",          # Logistic Regression
#                      "vglmContRatio",       # Logistic Regression
#                      "vglmCumulative",      # Logistic Regression
#                      
#                      "lssvmRadial",         # SVM
#                      "svmLinear",           # SVM
#                      "svmLinear2",          # SVM
#                      "svmRadialWeights",    # SVM
#                      "svmRadialSigma",      # SVM
#                      "svmRadialCost",       # SVM
#                      "svmRadial",           # SVM
#                      "svmPoly"   )          # SVM
#
#
#
#algo2 <- sort(algo[53:56], decreasing = T)
#algo3 <- algo[53:56]
#
#ensemble4 <- caretList(
#  reformulate(termlabels = colnames(train)[-1], response=colnames(train)[1]),
#  data=train, 
#  trControl=t, 
#  methodList=algo3, 
#  continue_on_fail=F,
#  metric="AUC"
#)
#library(caretEnsemble)
#library(caret)
#library(dplyr)
#library(here)
#library(pROC)
#t <- trainControl(
#  method = "repeatedcv", 
#  number = 10, 
#  index = createFolds(train[,1], 10),
#  repeats = 1, 
#  savePredictions = 'final', 
#  classProbs = TRUE, 
#  summaryFunction = multiClassSummary,
#  allowParallel = T, 
#  sampling = "up", 
#  returnResamp = "final",
#  selectionFunction = "best"
#)
#
#library(doParallel)
#cl <- makePSOCKcluster(3)
#registerDoParallel(cl)
#
### All subsequent models are then run in parallel
#model <- train(y ~ ., data = training, method = "rf")
#
### When you are done:
#stopCluster(cl)


##### 1st Step:
#multiclass <- function(tr, alg){
#  ensemble <- caretList(reformulate(termlabels = colnames(tr)[-1], response=colnames(tr)[1]),
#                        data=tr, 
#                        trControl=t, 
#                        methodList=alg, 
#                        continue_on_fail=T,
#                        metric="AUC")
#  model_preds <- lapply(ensemble, predict, newdata=test, type="raw")
#  x <- lapply(lapply(model_preds, as.numeric), multiclass.roc, response = as.numeric(test$vegetation.type))
#  resultado <- NA
#  for (i in 1:length(alg)){
#    resultado[i] <- as.numeric(x[[i]]$auc)
#  }
#  return(resultado)
#}
#
#result_1st_step2 <- replicate(2, multiclass(train2, algo[1:3]))
#result_1st_step
#auc_mean <- apply(result_1st_step, 1, mean)
#auc_sd   <- apply(result_1st_step, 1, sd)
#auc_max  <- apply(result_1st_step, 1, max)
#resultado_1st_step <- data.frame(algo = as.character(algo[1:3]), 
#                                 mean = as.numeric(auc_mean), 
#                                 sd   = as.numeric(auc_sd), 
#                                 max  = as.numeric(auc_max))

# Olhar para máximos e médias para identificar quais os melhores algoritmos.

####### Clustering ####
#nr <- createDataPartition(df[,1], p=0.9, list=FALSE)
#train <- df[nr,]
#test <- df[-nr,]
#
##train = clusters
#table(train$vegetation.type)
#
#rain <- train[train$vegetation.type == "Rainforest",-1]
#rest <- train[train$vegetation.type == "Restinga",-1]
#high <- train[train$vegetation.type == "High.Elevation",-1]
#semi <- train[train$vegetation.type == "Semideciduous",-1]
#rive <- train[train$vegetation.type == "Riverine",-1]
#
#rain <- kmeans(rain, 65)
#rest <- kmeans(rest, 65)
#high <- kmeans(high, 65)
#semi <- kmeans(semi, 65)
#rive <- kmeans(rive, 65)
#
#rain <- rain$centers
#rest <- rest$centers
#high <- high$centers
#semi <- semi$centers
#rive <- rive$centers
#
#vegetation.type <- c(rep("Rainforest", 65), 
#                     rep("Semideciduous", 65), 
#                     rep("High.Elevation", 65), 
#                     rep("Restinga", 65), 
#                     rep("Riverine", 65), 
#                     rep("Rocky", 65))
#train <- cbind(vegetation.type, rbind(rain, semi, high, rest, rive, train[train$vegetation.type == "Rocky",-1]))
#habitats <- levels(train[,1])
#table(train$vegetation.type)
#table(test$vegetation.type)


##t1: cluster / t2: upscaled
#t1 <- trainControl(
#  method = "repeatedcv", 
#  number = 10, 
#  index = createFolds(df[,1], 10),
#  repeats = 1000, 
#  savePredictions = 'final', 
#  classProbs = TRUE, 
#  summaryFunction = multiClassSummary,
#  allowParallel = T, 
#  returnResamp = "final",
#  selectionFunction = "best"
#)
#t2 <- trainControl(
#  method = "repeatedcv", 
#  number = 10, 
#  index = createFolds(train2[,1], 10),
#  repeats = 10, 
#  savePredictions = 'final', 
#  classProbs = TRUE, 
#  summaryFunction = multiClassSummary,
#  allowParallel = T, 
#  sampling = "up", 
#  returnResamp = "final",
#  selectionFunction = "best"
#)
#
## result1: clustered / result2: normal
#multiclass <- function(tr, te, alg, t){
#bayes <- c("manb", "naive_bayes", "nb", "nbDiscrete", "awnb", "nbSearch", "tan", "tanSearch", "awtan")
# ensemble <- caretList(reformulate(termlabels = colnames(train)[-1], response=colnames(train)[1]),
#                       data=df, 
#                       trControl=t1, 
#                       methodList=c("naive_bayes", "nnet", "mlp"), 
#                       continue_on_fail=T,
#                       metric="AUC")
#  model_preds <- lapply(ensemble, predict, newdata=te, type="raw")
#  x <- lapply(lapply(model_preds, as.numeric), multiclass.roc, response = as.numeric(te$vegetation.type))
#  resultado <- NA
#  for (i in 1:length(names(ensemble))){
#    resultado[i] <- as.numeric(x[[i]]$auc)
#  }
#  return(resultado)
#}
#result1 <- replicate(100, multiclass(train1, test1, algo, t1)) # clustered
#result2 <- replicate(1, multiclass(train2, test2, algo, t2)) # normal
#stopCluster(cl)
#
#tr <- train1
#te <- test1
#alg <- algo
#t <- t1
#
#auc_mean <- apply(result1, 1, mean)
#auc_sd   <- apply(result1, 1, sd)
#auc_max  <- apply(result1, 1, max)
#resultado1 <- data.frame(algo = as.character(algo), 
#                         mean = as.numeric(auc_mean), 
#                         sd   = as.numeric(auc_sd), 
#                         max  = as.numeric(auc_max))
#
##algos:
#"avNNet"           "dnn"              "mlp"              "mlpML"            "mlpWeightDecay"  
#[6] "mlpWeightDecayML" "monmlp"           "multinom"         "nnet"             "pcaNNet"         
#[11] "rbfDDA"           "cforest"          "extraTrees"       "parRF"            "ranger"          
#[16] "Rborist"          "rf"               "rFerns"           "RRF"              "RRFglobal"       
#[21] "wsrf"             "LMT"              "LogitBoost"       "polr"             "vglmAdjCat"      
#[26] "vglmContRatio"    "vglmCumulative"   "svmLinear"        "svmLinear2"       "svmRadialWeights"
#[31] "svmRadialSigma"   "svmRadialCost"    "svmRadial"        "svmPoly"  
#
#
#algo_2 <- c("mlpWeightDecayML",
#            "mlpML",
#            "mlpWeightDecay",
#            "mlp",
#            "nnet")
#result_2 <- replicate(1, multiclass(train1, test1, algo_2, t1)) # clustered

### Find bestTune for each algorithm:
#train_nnet <- train(vegetation.type ~ ., 
#                    method = "nnet", 
#                    data = train,
#                    tuneGrid = expand.grid(size = seq(1, 15, 1),
#                                          decay = seq(0,1,0.01)),
#                    trControl = trainControl(method = "cv", 
#                                             number = 10, 
#                                             p = .9,
#                                             savePredictions = 'final', 
#                                             classProbs = TRUE, 
#                                             summaryFunction = multiClassSummary,
#                                             allowParallel = T),
#                    metric="AUC"
#                    )
#preds_nnet <- predict(train_nnet, newdata=test, type="raw")
#roc_nnet <- multiclass.roc(as.numeric(preds_nnet), response = as.numeric(test$vegetation.type))
#roc_nnet$auc
#
#
#train_mlp1 <- train(vegetation.type ~ ., 
#                    method = "mlp", 
#                    data = train1,
#                    tuneGrid = expand.grid(size = seq(1, 10, 1)),
#                    trControl = trainControl(method = "cv", number = 10, p = .9)
#                    )
#
#train_mlp2 <- train(vegetation.type ~ ., 
#                    method = "mlpML", 
#                    data = train1,
#                    tuneGrid = expand.grid(size = seq(1, 10, 1),
#                                           decay = seq(0,1,0.01)),
#                    trControl = trainControl(method = "cv", number = 10, p = .9)
#                    )
#
#
#train_nb <- train(vegetation.type ~ ., 
#                    method = "naive_bayes", 
#                    data = train,
#                    tuneGrid = expand.grid(laplace = c(0),
#                                           usekernel = TRUE,
#                                           adjust = seq(0.01, 1, 0.02)),
#                  trControl = trainControl(method = "cv", 
#                                           number = 10, 
#                                           p = .9,
#                                           savePredictions = 'final', 
#                                           classProbs = TRUE, 
#                                           summaryFunction = multiClassSummary,
#                                           allowParallel = T),
#                  metric="AUC"
#)
#t1 <- trainControl(
#  method = "repeatedcv", 
#  number = 4, 
#  index = createFolds(train[,1], 4),
#  repeats = 10000, 
#  savePredictions = 'final', 
#  classProbs = TRUE, 
#  summaryFunction = multiClassSummary,
#  allowParallel = T, 
#  returnResamp = "final",
#  selectionFunction = "best"
#)
#ensemble2 <- caretList(reformulate(termlabels = colnames(train)[-1], response=colnames(train)[1]),
#                      data=train, 
#                      trControl=t1, 
#                      #methodList=c("naive_bayes", "nnet", "mlp"), 
#                      tuneList = list(
#                        naive_bayes = caretModelSpec(method = "naive_bayes", 
#                                                     tuneGrid = expand.grid(laplace = c(0),
#                                                                            usekernel = TRUE,
#                                                                            adjust = seq(0.01, 1, 0.02))),
#                        nnet = caretModelSpec(method = "nnet", 
#                                              tuneGrid = expand.grid(size = seq(1, 15, 1),
#                                                                     decay = seq(0,1,0.01))),
#                        mlp = caretModelSpec(method = "mlp", 
#                                             tuneGrid = expand.grid(size = seq(1, 10, 1)))
#                                      ),
#                      continue_on_fail=T,
#                      metric="AUC")



