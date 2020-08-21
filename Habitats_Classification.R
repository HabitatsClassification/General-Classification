library(caretEnsemble)
library(caret)
library(dplyr)

#### Create/Open Data ####
algo <- c("kknn","rf", "LogitBoost", "nnet", "svmRadial")
algo <- c("RRF", "ranger", #"ORFridge", # Random Forest 
                     "LogitBoost","vglmContRatio", "vglmCumulative", # Logistic Regression
                     "svmRadialCost", "svmRadialSigma", "svmRadial", #SVM
                     "pcaNNet", "avNNet", "rbfDDA",
                     "kknn") # NeuralNetwork
df <- read.csv("NTT_xgboost.csv")
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

classification <- function(df){
  nr <- createDataPartition(df$vegetation.type, p=0.9, list=FALSE)
  train <- df[nr,]
  test <- df[-nr,]
  #### Generatind Methods ####
  multiclass <- function(train, test, algo, th){
    t <- trainControl(method = "repeatedcv", number = 10, index = createFolds(train$vegetation.type, 10),
                      repeats = 10, savePredictions = 'final', classProbs = TRUE, summaryFunction = multiClassSummary,
                      allowParallel = F, sampling = "up", returnResamp = "final")
    ensemble <- caretList(vegetation.type ~ ., data=train, trControl=t, methodList=algo, continue_on_fail=T)
    model_preds <- lapply(ensemble, predict, newdata=test, type="prob")
    
    auc_knn  <- ensemble$kknn$results$AUC[ensemble$kknn$results$kmax == ensemble$kknn$bestTune$kmax]
    auc_rf   <- ensemble$rf$results$AUC[ensemble$rf$results$mtry == ensemble$rf$bestTune[1,]]
    auc_log  <- ensemble$LogitBoost$results$AUC[ensemble$LogitBoost$results$nIter == ensemble$LogitBoost$bestTune[1,]]
    auc_nnet <- ensemble$nnet$results$AUC[ensemble$nnet$results$size == ensemble$nnet$bestTune[1,1] & 
                                            ensemble$nnet$results$decay == ensemble$nnet$bestTune[1,2]]
    auc_svm  <- ensemble$svmRadial$results$AUC[ensemble$svmRadial$results$sigma == ensemble$svmRadial$bestTune[1,1] &
                                                 ensemble$svmRadial$results$C == ensemble$svmRadial$bestTune[1,2] ]
    
    if(auc_knn  < th){ auc_knn  <- 0}
    if(auc_rf   < th){ auc_rf   <- 0}
    if(auc_log  < th){ auc_log  <- 0}
    if(auc_nnet < th){ auc_nnet <- 0}
    if(auc_svm  < th){ auc_svm  <- 0}
    auc_sum <- auc_knn + auc_rf + auc_log + auc_nnet + auc_svm 
    
    Restinga_wmean <- (model_preds$kknn$Restinga*auc_knn + model_preds$svm$Restinga*auc_svm +
                         model_preds$rf$Restinga*auc_rf + model_preds$LogitBoost$Restinga*auc_log +   
                         model_preds$nnet$Restinga*auc_rf) / auc_sum
    Rainforest_wmean <- (model_preds$kknn$Rainforest*auc_knn + model_preds$svm$Rainforest*auc_svm +
                           model_preds$rf$Rainforest*auc_rf + model_preds$LogitBoost$Rainforest*auc_log +   
                           model_preds$nnet$Rainforest*auc_rf) / auc_sum
    Semideciduous_wmean <- (model_preds$kknn$Semideciduous*auc_knn + model_preds$svm$Semideciduous*auc_svm +
                              model_preds$rf$Semideciduous*auc_rf + model_preds$LogitBoost$Semideciduous*auc_log +   
                              model_preds$nnet$Semideciduous*auc_rf) / auc_sum
    High.Elevation_wmean <- (model_preds$kknn$High.Elevation*auc_knn + model_preds$svm$High.Elevation*auc_svm +
                               model_preds$rf$High.Elevation*auc_rf + model_preds$LogitBoost$High.Elevation*auc_log +   
                               model_preds$nnet$High.Elevation*auc_rf) / auc_sum
    Rocky_wmean <- (model_preds$kknn$Rocky*auc_knn + model_preds$svm$Rocky*auc_svm +
                      model_preds$rf$Rocky*auc_rf + model_preds$LogitBoost$Rocky*auc_log +   
                      model_preds$nnet$Rocky*auc_rf) / auc_sum
    Riverine_wmean <- (model_preds$kknn$Riverine*auc_knn + model_preds$svm$Riverine*auc_svm +
                         model_preds$rf$Riverine*auc_rf + model_preds$LogitBoost$Riverine*auc_log +   
                         model_preds$nnet$Riverine*auc_rf) / auc_sum
    result <- cbind(test, Rainforest_wmean, Semideciduous_wmean, High.Elevation_wmean, Restinga_wmean, Riverine_wmean, Rocky_wmean)
    final_prediction <- apply(result[,-(1:4)],1,function(x) gsub("_wmean","",names(which.max(x))) )
    roc_probs <- NA
    for(i in 1:nrow(result)){
      roc_probs[i] <- result[i,colnames(result) == paste0(result$vegetation.type, "_wmean")[i]]
    }
    result <- cbind(result, final_prediction, roc_probs)
    return(result)
  }
  one_vs_all <- function(train, test, algo){
    Rainforest <- train
    Rainforest$vegetation.type <- ifelse(Rainforest$vegetation.type != "Rainforest", "Other", "Rainforest") %>% 
      factor(levels = c("Rainforest", "Other"))
    High.Elevation <- train
    High.Elevation$vegetation.type <- ifelse(High.Elevation$vegetation.type != "High.Elevation", "Other", "High.Elevation") %>% 
      factor(levels = c("High.Elevation", "Other"))
    Semideciduous <- train
    Semideciduous$vegetation.type <- ifelse(Semideciduous$vegetation.type != "Semideciduous", "Other", "Semideciduous") %>% 
      factor(levels = c("Semideciduous", "Other"))
    Restinga <- train
    Restinga$vegetation.type <- ifelse(Restinga$vegetation.type != "Restinga", "Other", "Restinga") %>% 
      factor(levels = c("Restinga", "Other"))
    Riverine <- train
    Riverine$vegetation.type <- ifelse(Riverine$vegetation.type != "Riverine", "Other", "Riverine") %>% 
      factor(levels = c("Riverine", "Other"))
    Rocky <- train
    Rocky$vegetation.type <- ifelse(Rocky$vegetation.type != "Rocky", "Other", "Rocky") %>% 
      factor(levels = c("Rocky", "Other"))
    
    habitat_classification <- function(df, algo){
      nr <- createDataPartition(df$vegetation.type, p=0.9, list=FALSE)
      train2 <- df[nr,]
      u <- unique(df$vegetation.type)
      u <- u[u != "Other"]
      t <- trainControl(method = "repeatedcv", number = 10, index = createFolds(train2$vegetation.type, 10),
                        repeats = 10, savePredictions = 'final', classProbs = TRUE, summaryFunction = twoClassSummary,
                        allowParallel = F, sampling = "down")
      ensemble <-  caretList(vegetation.type ~ ., data=train2, trControl=t, methodList=algo, continue_on_fail=T,  verbose=T)
      assign(paste0(u,"_",algo[1]), ensemble)
    }
    
    # Run Function:
    Rainforest_result     <- habitat_classification(Rainforest     , algo)
    Riverine_result       <- habitat_classification(Riverine       , algo)
    Rocky_result          <- habitat_classification(Rocky          , algo)
    Restinga_result       <- habitat_classification(Restinga       , algo)
    Semideciduous_result  <- habitat_classification(Semideciduous  , algo)
    High.Elevation_result <- habitat_classification(High.Elevation , algo)
    
    ensembles <- c("Rainforest_result",
                   "Riverine_result",
                   "Rocky_result",
                   "Restinga_result",
                   "Semideciduous_result",
                   "High.Elevation_result")
    
    for (j in 1:length(ensembles)){
      print(j)
      en <- get(ensembles[j])
      
      ROC_knn  <- en$kknn$results$ROC[en$kknn$results$kmax == en$kknn$bestTune$kmax]
      ROC_rf   <- en$rf$results$ROC[en$rf$results$mtry == en$rf$bestTune[1,]]
      ROC_log  <- en$LogitBoost$results$ROC[en$LogitBoost$results$nIter == en$LogitBoost$bestTune[1,]]
      ROC_nnet <- en$nnet$results$ROC[en$nnet$results$size == en$nnet$bestTune[1,1] & 
                                        en$nnet$results$decay == en$nnet$bestTune[1,2]]
      ROC_svm  <- en$svmRadial$results$ROC[en$svmRadial$results$sigma == en$svmRadial$bestTune[1,1] &
                                             en$svmRadial$results$C == en$svmRadial$bestTune[1,2] ]
      
      if(ROC_knn  < 0.6){ ROC_knn  <- 0}
      if(ROC_rf   < 0.6){ ROC_rf   <- 0}
      if(ROC_log  < 0.6){ ROC_log  <- 0}
      if(ROC_nnet < 0.6){ ROC_nnet <- 0}
      if(ROC_svm  < 0.6){ ROC_svm  <- 0}
      
      ROC_sum <- ROC_knn + ROC_rf + ROC_log + ROC_nnet + ROC_svm 
      
      model_preds <- lapply(en, predict, newdata=test, type="prob")
      
      assign(paste0(gsub("_result", "_wmean", ensembles[j])), 
             (model_preds$kknn[model_preds$kknn != "Other"]*ROC_knn +
                model_preds$svmRadial[model_preds$svmRadial != "Other"]*ROC_svm +
                model_preds$rf[model_preds$rf != "Other"]*ROC_rf +
                model_preds$LogitBoost[model_preds$LogitBoost != "Other"]*ROC_log +  
                model_preds$nnet[model_preds$nnet != "Other"]*ROC_nnet) / ROC_sum )
    }
    
    result <- cbind(test, Rainforest_wmean, Semideciduous_wmean, High.Elevation_wmean, Restinga_wmean, Riverine_wmean, Rocky_wmean)
    
    final_prediction <- colnames(result[,-(1:4)])[apply(result[,-(1:4)],1,which.max)]
    final_prediction <- gsub("_wmean", "", final_prediction)
    auc_probs <- NA
    for(i in 1:nrow(result)){
      auc_probs[i] <- result[i,colnames(result) == paste0(result$vegetation.type, "_wmean")[i]]
    }
    result <- cbind(result, final_prediction, auc_probs)
    return(result)
  }
  result_multiclass <- multiclass(train, test, algo)
  result_one_vs_all <- one_vs_all(train, test, algo)
  #### Comparing Methods ####
  mc.roc <- multiclass.roc(result_multiclass$vegetation.type, result_multiclass$roc_probs)
  oa.roc <- multiclass.roc(result_one_vs_all$vegetation.type, result_one_vs_all$auc_probs)
  result <- list("Multiclass" = mc.roc, "One.vs.all" = oa.roc)
  return(result)
}

n <- 100
result_multiclass_100_newalgo <- replicate(n, classification(df))

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
