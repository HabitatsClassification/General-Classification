library(caretEnsemble)
library(caret)
library(dplyr)
library(here)
library(pROC)

#### Create/Open Data ####
algo <- c("kknn","rf", "LogitBoost", "nnet", "svmRadial")
algo <- c("RRF", "ranger", #"ORFridge", # Random Forest 
                     "LogitBoost","vglmContRatio", "vglmCumulative", # Logistic Regression
                     "svmRadialCost", "svmRadialSigma", "svmRadial", #SVM
                     "pcaNNet", "avNNet", "rbfDDA",
                     "kknn") # NeuralNetwork
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
  multiclass <- function(train, test, algo){
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
        vegetation.type ~ ., # transformar "vegetation.type" em algo mais genérico
        data=train, 
        trControl=t, 
        methodList=algo, 
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
      df2 <- Map('*',df2,values_auc)
      df2 <- as.data.frame(df2)
      df2 <- rowSums(df2)/sum_auc
      result <- cbind(result,df2)
    }
    colnames(result) <- c(colnames(test), habitats)
    final_prediction <- apply(result[,-(1:4)],1,function(x) names(which.max(x)))
    habitat_probs <- NA
    for(i in 1:nrow(result)){
      habitat_probs[i] <- result[i,colnames(result) == result[,1][i]]
    }
    result <- cbind(result, final_prediction, habitat_probs)
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
    
    habitat_classification <- function(df, algo, th=0.6){
      nr <- createDataPartition(df$vegetation.type, p=0.9, list=FALSE)
      train2 <- df[nr,]
      u <- unique(df$vegetation.type)
      u <- u[u != "Other"]
      t <- trainControl(method = "repeatedcv", 
                        number = 10, 
                        index = createFolds(train2$vegetation.type, 10),
                        repeats = 10, 
                        savePredictions = 'final', 
                        classProbs = TRUE, 
                        summaryFunction = twoClassSummary,
                        allowParallel = F, 
                        sampling = "down")
      ensemble <-  caretList(vegetation.type ~ ., 
                             data=train2, 
                             trControl=t, 
                             methodList=algo, 
                             continue_on_fail=T,
                             metric="ROC",
                             verbose=T)
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
    ensembles2 <- c("Rainforest",
                    "Riverine",
                    "Rocky",
                    "Restinga",
                    "Semideciduous",
                    "High.Elevation")
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
      df2 <- sapply(model_preds,'[',ensembles2[j])
      df2 <- Map('*',df2,values_roc)
      df2 <- as.data.frame(df2)
      df2 <- rowSums(df2)/sum_roc
      result <- cbind(result,df2)
    }
    
    colnames(result) <- c(colnames(test), ensembles2)
    
    
    final_prediction <- colnames(result[,-(1:4)])[apply(result[,-(1:4)],1,which.max)]
    habitat_probs <- NA
    for(i in 1:nrow(result)){
      habitat_probs[i] <- result[i,colnames(result) == result[,1][i]]
    }
    result <- cbind(result, final_prediction, habitat_probs)
    return(result)
  }
  result_multiclass <- multiclass(train, test, algo)
  result_one_vs_all <- one_vs_all(train, test, algo)
  #### Comparing Methods ####
  mc.roc <- multiclass.roc(result_multiclass[,1], result_multiclass$habitat_probs)
  oa.roc <- multiclass.roc(result_one_vs_all[,1], result_one_vs_all$habitat_probs)
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
