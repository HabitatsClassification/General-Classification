## ================================================================================================================== ##
##                                                                                                                    ##
##           Applying Machine Learning Algorithms for Habitats Classification from Atlantic Rainforest                ##
##                                                                                                                    ##
## ================================================================================================================== ##
##                                                                                                                    ##
## Author: MSc. Luíz Fernando Esser                                                                                   ##
##                                                                                                                    ##
## Date Created: October, 11th, 2020                                                                                  ##
##                                                                                                                    ##
## This work is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.   ##
## To view a copy of this license, visit http://creativecommons.org/licenses/by-nc-sa/4.0/                            ##
## or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.                                   ##
##                                                                                                                    ##
## Email: luizesser@gmail.com                                                                                         ##
##                                                                                                                    ##
## ================================================================================================================== ##
##                                                                                                                    ##
## Notes:                                                                                                             ##
##                                                                                                                    ##
##                                                                                                                    ##
## ================================================================================================================== ##

# Open necessary libraries:
library(caretEnsemble)
library(caret)
library(dplyr)
library(here)
library(pROC)
library(dplyr)

# Open NeoTropTree dataframe:
df <- read.csv(here("NTT_habitats_classification.csv"))
df <- df[,-1]
head(df)
summary(df)
#

# First Step: Framework Selection ####
algo <- c("nb","rf", "LogitBoost", "nnet", "svmRadial")

classification <- function(df, th = 0.5){
  habitats <- levels(df[,1])
  nr <- createDataPartition(df[,1], p=0.9, list=FALSE)
  train <- df[nr,]
  test <- df[-nr,]
  #### Generatind Methods ####
  multiclass <- function(train, test, algo, th){
    t <- trainControl(
        method = "repeatedcv", 
        number = 10, 
        index = createFolds(train[,1], 10, returnTrain = T),
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
        methodList=algo, 
        continue_on_fail=T,
        metric="AUC"
      )
    model_preds <- lapply(ensemble, predict, newdata=test, type="prob")
    
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
                        index = createFolds(df_habitat[,1], 10, returnTrain = T),
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
  result_multiclass <- multiclass(train, test, algo, th)
  result_one_vs_all <- one_vs_all(train, test, algo, th)
  #### Comparing Methods ####
  mc.roc <- multiclass.roc(result_multiclass[,1], result_multiclass$habitat_probs)
  oa.roc <- multiclass.roc(result_one_vs_all[,1], result_one_vs_all$habitat_probs)
  result <- list("Multiclass" = as.numeric(mc.roc$auc), "One.vs.all" = as.numeric(oa.roc$auc))
  return(result)
}

n <- 100
result_framework <- replicate(n, classification(df, 0.5))
saveRDS(result_framework, "result_framework.rds")
df_new <- data.frame(multiclass = unlist(result_framework[1,]), 
                     one_vs_all = unlist(result_framework[2,]))
df_ttest <- data.frame( method = c(rep("multiclass", n), rep("one_vs_all", n)),
                        AUC = c(df_new$multiclass, df_new$one_vs_all))
t_st <- t.test(AUC ~ method, df_ttest)
mean(df_new$multiclass)
mean(df_new$one_vs_all)
sd(df_new$multiclass)
sd(df_new$one_vs_all)

boxplot(AUC ~ method, data=df_ttest,
        col=(c("gray","white")),
        xlab="Framework")
ggplot(df_ttest, aes(AUC, fill=method)) + geom_density(alpha = 0.4)



#Welch Two Sample t-test
#
#data:  AUC by method
#t = 6.8179, df = 197.98, p-value = 1.091e-10
#alternative hypothesis: true difference in means is not equal to 0
#95 percent confidence interval:
#  0.02172926 0.03941441
#sample estimates:
#  mean in group multiclass mean in group one_vs_all 
#0.6150572                0.5844854



# Second Step: Selecting Algorithms ####
algo <- c(# NeuralNetwork
          "avNNet", "dnn", "mlp", "mlpML", "mlpWeightDecay",
          "mlpWeightDecayML", "monmlp", "multinom", "nnet",
          "pcaNNet", "rbfDDA",
          # Random Forest
          "cforest", "ranger", "Rborist",
          "rf", "RRF", "RRFglobal", "wsrf",                
          # Logistic Regression
          "LMT", "LogitBoost", "polr", "vglmAdjCat", "vglmContRatio", "vglmCumulative", 
          # SVM
          "lssvmRadial", "svmLinear", "svmLinear2", "svmRadialWeights",
          "svmRadialSigma", "svmRadialCost", "svmRadial", "svmPoly", 
          # Naive Bayes
          "nb", "naive_bayes")

multiclass <- function(n,m){
  resultado <- list()
  for(j in 1:n){
    print(j)
    nr <- createDataPartition(df[,1], p=0.9, list=FALSE)
    train <- df[nr,]
    test <- df[-nr,]
    t <- trainControl(
      method = "repeatedcv", 
      number = 10, 
      index = createFolds(train[,1], 10, returnTrain = T),
      repeats = 10, 
      savePredictions = 'final', 
      classProbs = TRUE, 
      summaryFunction = multiClassSummary,
      allowParallel = T, 
      sampling = "up", 
      returnResamp = "final",
      selectionFunction = "best"
    )
    ensemble <- caretList(
      reformulate(termlabels = colnames(train)[-1], response=colnames(train)[1]),
      data=train, 
      trControl=t, 
      methodList=algo[m], 
      continue_on_fail=T,
      metric="AUC"
    )
    model_preds <- lapply(ensemble, predict, newdata=test, type="raw")
    x <- lapply(lapply(model_preds, as.numeric), multiclass.roc, response = as.numeric(test$vegetation.type))
    preds_auc <- NA
    for (i in 1:length(names(ensemble))){
      preds_auc[i] <- as.numeric(x[[i]]$auc)
    }
    resultado[[j]] <- data.frame(algo = names(ensemble),
                                 auc = preds_auc)
  }
  return(resultado)
}

result_multiclass <- multiclass(100)
result_df <- bind_rows(result_multiclass)
bmean <- aggregate(result_df$auc, by=list(Category = result_df$algo), FUN=mean)
bmax <- aggregate(result_df$auc, by=list(Category = result_df$algo), FUN=max)

result_aov <- aov(auc ~ algo, result_df)
summary(result_aov)
tuk <- TukeyHSD(result_aov)
tuk <- as.data.frame(tuk$algo)

# Third Step: Instance Selection ####
instance_selection <- function(df, n_clust = 100, n_clust2 = 50){
  nr <- createDataPartition(df[,1], p=0.9, list=FALSE)
  train <- df[nr,]
  test <- df[-nr,]
  # train_IS1: uses 50 clusters centroides ####
  print(paste0("Train IS1 Started."))
  rain <- train[train$vegetation.type == "Rainforest",-1]
  rest <- train[train$vegetation.type == "Restinga",-1]
  high <- train[train$vegetation.type == "High.Elevation",-1]
  semi <- train[train$vegetation.type == "Semideciduous",-1]
  rive <- train[train$vegetation.type == "Riverine",-1]
  rock <- train[train$vegetation.type == "Rocky",-1]
  rain <- kmeans(rain, n_clust)
  rest <- kmeans(rest, n_clust)
  high <- kmeans(high, n_clust)
  semi <- kmeans(semi, n_clust)
  rive <- kmeans(rive, n_clust)
  rock <- kmeans(rock, n_clust)
  rain <- rain$centers
  rest <- rest$centers
  high <- high$centers
  semi <- semi$centers
  rive <- rive$centers
  rock <- rock$centers
  vegetation.type <- c(rep("Rainforest",     n_clust), 
                       rep("Semideciduous",  n_clust), 
                       rep("High.Elevation", n_clust), 
                       rep("Restinga",       n_clust), 
                       rep("Riverine",       n_clust), 
                       rep("Rocky",          n_clust))
  families <- as.data.frame(rbind(rain, semi, high, rest, rive, rock))
  train_IS1 <- data.frame(vegetation.type = as.character(vegetation.type),
                      myrtaceae = as.numeric(families$myrtaceae),
                      fabaceae  = as.numeric(families$fabaceae),
                      rubiaceae = as.numeric(families$rubiaceae))
  
  # train_IS2: removes outliers from 50 clusters, scales up ####
  print(paste0("Train IS2 Started."))
  rain <- train[train$vegetation.type == "Rainforest",-1]
  rest <- train[train$vegetation.type == "Restinga",-1]
  high <- train[train$vegetation.type == "High.Elevation",-1]
  semi <- train[train$vegetation.type == "Semideciduous",-1]
  rive <- train[train$vegetation.type == "Riverine",-1]
  rock <- train[train$vegetation.type == "Rocky",-1]
  rain_clust <- kmeans(rain, n_clust)
  rest_clust <- kmeans(rest, n_clust)
  high_clust <- kmeans(high, n_clust)
  semi_clust <- kmeans(semi, n_clust)
  rive_clust <- kmeans(rive, n_clust)
  rock_clust <- kmeans(rock, n_clust)
  for (i in 1:n_clust){
  df_clust <- rain[rain_clust$cluster == i,-4]
  x <- sort(as.matrix(dist(rbind(rain_clust$centers[i,],df_clust)))[,1])[-1]
  if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
  x <- x[1:num]
  df_clust <- df[names(x),]
  if(i == 1){train_rain <- df_clust} else { train_rain <- rbind(train_rain, df_clust)}
} # rain
  for (i in 1:n_clust){
  df_clust <- rest[rest_clust$cluster == i,-4]
  x <- sort(as.matrix(dist(rbind(rest_clust$centers[i,],df_clust)))[,1])[-1]
  if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
  x <- x[1:num]
  df_clust <- df[names(x),]
  if(i == 1){train_rest<- df_clust} else { train_rest <- rbind(train_rest, df_clust)}
} # rest
  for (i in 1:n_clust){
  df_clust <- high[high_clust$cluster == i,-4]
  x <- sort(as.matrix(dist(rbind(high_clust$centers[i,],df_clust)))[,1])[-1]
  if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
  x <- x[1:num]
  df_clust <- df[names(x),]
  if(i == 1){train_high <- df_clust} else { train_high <- rbind(train_high, df_clust)}
} # high
  for (i in 1:n_clust){
  df_clust <- semi[semi_clust$cluster == i,-4]
  x <- sort(as.matrix(dist(rbind(semi_clust$centers[i,],df_clust)))[,1])[-1]
  if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
  x <- x[1:num]
  df_clust <- df[names(x),]
  if(i == 1){train_semi <- df_clust} else { train_semi <- rbind(train_semi, df_clust)}
} # semi
  for (i in 1:n_clust){
  df_clust <- rive[rive_clust$cluster == i,-4]
  x <- sort(as.matrix(dist(rbind(rive_clust$centers[i,],df_clust)))[,1])[-1]
  if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
  x <- x[1:num]
  df_clust <- df[names(x),]
  if(i == 1){train_rive <- df_clust} else { train_rive <- rbind(train_rive, df_clust)}
} # rive
  for (i in 1:n_clust){
  df_clust <- rock[rock_clust$cluster == i,-4]
  x <- sort(as.matrix(dist(rbind(rock_clust$centers[i,],df_clust)))[,1])[-1]
  if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
  x <- x[1:num]
  df_clust <- df[names(x),]
  if(i == 1){train_rock <- df_clust} else { train_rock <- rbind(train_rock, df_clust)}
} # rock
  train_IS2 <- rbind(train_rain, train_rest, train_high, train_semi, train_rive, train_rock)

  # train_IS3: removes outliers from one cluster, scales up ####
  print(paste0("Train IS3 Started."))
  n_clust <- 1
  rain <- train[train$vegetation.type == "Rainforest",-1]
  rest <- train[train$vegetation.type == "Restinga",-1]
  high <- train[train$vegetation.type == "High.Elevation",-1]
  semi <- train[train$vegetation.type == "Semideciduous",-1]
  rive <- train[train$vegetation.type == "Riverine",-1]
  rock <- train[train$vegetation.type == "Rocky",-1]
  rain_clust <- kmeans(rain, 1)
  rest_clust <- kmeans(rest, 1)
  high_clust <- kmeans(high, 1)
  semi_clust <- kmeans(semi, 1)
  rive_clust <- kmeans(rive, 1)
  rock_clust <- kmeans(rock, 1)
  for (i in 1:1){
    df_clust <- rain[rain_clust$cluster == i,-4]
    x <- sort(as.matrix(dist(rbind(rain_clust$centers[i,],df_clust)))[,1])[-1]
    if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
    x <- x[1:num]
    df_clust <- df[names(x),]
    if(i == 1){train_rain <- df_clust} else { train_rain <- rbind(train_rain, df_clust)}
  } # rain
  for (i in 1:1){
    df_clust <- rest[rest_clust$cluster == i,-4]
    x <- sort(as.matrix(dist(rbind(rest_clust$centers[i,],df_clust)))[,1])[-1]
    if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
    x <- x[1:num]
    df_clust <- df[names(x),]
    if(i == 1){train_rest<- df_clust} else { train_rest <- rbind(train_rest, df_clust)}
  } # rest
  for (i in 1:1){
    df_clust <- high[high_clust$cluster == i,-4]
    x <- sort(as.matrix(dist(rbind(high_clust$centers[i,],df_clust)))[,1])[-1]
    if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
    x <- x[1:num]
    df_clust <- df[names(x),]
    if(i == 1){train_high <- df_clust} else { train_high <- rbind(train_high, df_clust)}
  } # high
  for (i in 1:1){
    df_clust <- semi[semi_clust$cluster == i,-4]
    x <- sort(as.matrix(dist(rbind(semi_clust$centers[i,],df_clust)))[,1])[-1]
    if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
    x <- x[1:num]
    df_clust <- df[names(x),]
    if(i == 1){train_semi <- df_clust} else { train_semi <- rbind(train_semi, df_clust)}
  } # semi
  for (i in 1:1){
    df_clust <- rive[rive_clust$cluster == i,-4]
    x <- sort(as.matrix(dist(rbind(rive_clust$centers[i,],df_clust)))[,1])[-1]
    if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
    x <- x[1:num]
    df_clust <- df[names(x),]
    if(i == 1){train_rive <- df_clust} else { train_rive <- rbind(train_rive, df_clust)}
  } # rive
  for (i in 1:1){
    df_clust <- rock[rock_clust$cluster == i,-4]
    x <- sort(as.matrix(dist(rbind(rock_clust$centers[i,],df_clust)))[,1])[-1]
    if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
    x <- x[1:num]
    df_clust <- df[names(x),]
    if(i == 1){train_rock <- df_clust} else { train_rock <- rbind(train_rock, df_clust)}
  } # rock
  train_IS3 <- rbind(train_rain, train_rest, train_high, train_semi, train_rive, train_rock)

  # train_IS4: removes outliers from one cluster and make 30 clusters out of the result ####
  print(paste0("Train IS4 Started."))
  rain <- train[train$vegetation.type == "Rainforest",-1]
  rest <- train[train$vegetation.type == "Restinga",-1]
  high <- train[train$vegetation.type == "High.Elevation",-1]
  semi <- train[train$vegetation.type == "Semideciduous",-1]
  rive <- train[train$vegetation.type == "Riverine",-1]
  rock <- train[train$vegetation.type == "Rocky",-1]
  rain_clust <- kmeans(rain, 1)
  rest_clust <- kmeans(rest, 1)
  high_clust <- kmeans(high, 1)
  semi_clust <- kmeans(semi, 1)
  rive_clust <- kmeans(rive, 1)
  rock_clust <- kmeans(rock, 1)
  for (i in 1:1){
    df_clust <- rain[rain_clust$cluster == i,-4]
    x <- sort(as.matrix(dist(rbind(rain_clust$centers[i,],df_clust)))[,1])[-1]
    if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
    x <- x[1:num]
    df_clust <- df[names(x),]
    if(i == 1){train_rain <- df_clust} else { train_rain <- rbind(train_rain, df_clust)}
  } # rain
  for (i in 1:1){
    df_clust <- rest[rest_clust$cluster == i,-4]
    x <- sort(as.matrix(dist(rbind(rest_clust$centers[i,],df_clust)))[,1])[-1]
    if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
    x <- x[1:num]
    df_clust <- df[names(x),]
    if(i == 1){train_rest<- df_clust} else { train_rest <- rbind(train_rest, df_clust)}
  } # rest
  for (i in 1:1){
    df_clust <- high[high_clust$cluster == i,-4]
    x <- sort(as.matrix(dist(rbind(high_clust$centers[i,],df_clust)))[,1])[-1]
    if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
    x <- x[1:num]
    df_clust <- df[names(x),]
    if(i == 1){train_high <- df_clust} else { train_high <- rbind(train_high, df_clust)}
  } # high
  for (i in 1:1){
    df_clust <- semi[semi_clust$cluster == i,-4]
    x <- sort(as.matrix(dist(rbind(semi_clust$centers[i,],df_clust)))[,1])[-1]
    if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
    x <- x[1:num]
    df_clust <- df[names(x),]
    if(i == 1){train_semi <- df_clust} else { train_semi <- rbind(train_semi, df_clust)}
  } # semi
  for (i in 1:1){
    df_clust <- rive[rive_clust$cluster == i,-4]
    x <- sort(as.matrix(dist(rbind(rive_clust$centers[i,],df_clust)))[,1])[-1]
    if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
    x <- x[1:num]
    df_clust <- df[names(x),]
    if(i == 1){train_rive <- df_clust} else { train_rive <- rbind(train_rive, df_clust)}
  } # rive
  for (i in 1:1){
    df_clust <- rock[rock_clust$cluster == i,-4]
    x <- sort(as.matrix(dist(rbind(rock_clust$centers[i,],df_clust)))[,1])[-1]
    if((length(x) %% 2) == 0){num <- length(x)/2} else {num <- length(x)/2+1}
    x <- x[1:num]
    df_clust <- df[names(x),]
    if(i == 1){train_rock <- df_clust} else { train_rock <- rbind(train_rock, df_clust)}
  } # rock
  train_IS4 <- rbind(train_rain, train_rest, train_high, train_semi, train_rive, train_rock)
  # build n clusters:
  rain <- train_IS4[train_IS4$vegetation.type == "Rainforest",-1]
  rest <- train_IS4[train_IS4$vegetation.type == "Restinga",-1]
  high <- train_IS4[train_IS4$vegetation.type == "High.Elevation",-1]
  semi <- train_IS4[train_IS4$vegetation.type == "Semideciduous",-1]
  rive <- train_IS4[train_IS4$vegetation.type == "Riverine",-1]
  rock <- train_IS4[train_IS4$vegetation.type == "Rocky",-1]
  rain <- kmeans(rain, n_clust2)
  rest <- kmeans(rest, n_clust2)
  high <- kmeans(high, n_clust2)
  semi <- kmeans(semi, n_clust2)
  rive <- kmeans(rive, n_clust2)
  rock <- kmeans(rock, n_clust2)
  rain <- rain$centers
  rest <- rest$centers
  high <- high$centers
  semi <- semi$centers
  rive <- rive$centers
  rock <- rock$centers
  vegetation.type <- c(rep("Rainforest",     n_clust2), 
                       rep("Semideciduous",  n_clust2), 
                       rep("High.Elevation", n_clust2), 
                       rep("Restinga",       n_clust2), 
                       rep("Riverine",       n_clust2), 
                       rep("Rocky",          n_clust2))
  families <- as.data.frame(rbind(rain, semi, high, rest, rive, rock))
  train_IS4 <- data.frame(vegetation.type = as.character(vegetation.type),
                      myrtaceae = as.numeric(families$myrtaceae),
                      fabaceae  = as.numeric(families$fabaceae),
                      rubiaceae = as.numeric(families$rubiaceae))
    
  # Finnished Train Data Building ####
  algo <- c("cforest", "ranger", "LogitBoost")
  multiclass <- function(tr, th = 0.5){
    habitats <- levels(df[,1])
    t <- trainControl(
      method = "repeatedcv", 
      number = 10, 
      index = createFolds(tr[,1], 10, returnTrain = T),
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
      reformulate(termlabels = colnames(tr)[-1], response=colnames(tr)[1]),
      data=tr, 
      trControl=t, 
      methodList=algo, 
      continue_on_fail=T,
      metric="AUC"
    )
    model_preds <- lapply(ensemble, predict, newdata=test, type="prob")
    
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
    final_prediction <- as.data.frame(apply(result[,-(1:ncol(df))],1,function(x) names(which.max(x))))
    x <- multiclass.roc(response = as.numeric(final_prediction[,1]), predictor = as.numeric(test[,1]))
    x <- as.numeric(x$auc)
  return(x)
  }
  train_data <- list(train_IS1,train_IS2,train_IS3,train_IS4)
  print(paste0("Applying data to multiclass function"))
  z <- mapply(multiclass, train_data)
  z2 <- data.frame(IS = c("IS1","IS2","IS3","IS4"), z)
  print(paste0("Instance Selection Finnished"))
  return(z)
}
n <- 100
set.seed(1)
result_instance_selection <- replicate(n, instance_selection(df))
result_instance_selection <- as.data.frame(result_instance_selection)
rownames(result_instance_selection) <- c("IS1","IS2","IS3","IS4")
df_aov <- data.frame( method = c(rep("IS1", n),
                                 rep("IS2", n),
                                 rep("IS3", n),
                                 rep("IS4", n)),
                        AUC = c(as.numeric(result_instance_selection[1,]),
                                as.numeric(result_instance_selection[2,]),
                                as.numeric(result_instance_selection[3,]),
                                as.numeric(result_instance_selection[4,])))
result_aov <- aov(AUC ~ method, df_aov)
summary(result_aov)
summary.lm(result_aov)
boxplot(df_aov$AUC~df_aov$method)
tuk <- TukeyHSD(result_aov)

ggplot(df_aov, aes(x = method, y = AUC, fill=method)) + 
  geom_boxplot() + 
  labs(x = "Method") +
  theme(legend.position="none")

# Fourth Step: Algorithms Tuning ####
set.seed(1)
nr <- createDataPartition(df[,1], p=0.9, list=FALSE)
train <- df[nr,]
test <- df[-nr,]
set.seed(1)
ensemble <- caretList(reformulate(termlabels = colnames(train)[-1], response=colnames(train)[1]),
                      data=train,
                      trControl=trainControl(
                        method = "repeatedcv", 
                        number = 10, 
                        index = createFolds(train[,1], 10, returnTrain = T),
                        repeats = 10, 
                        savePredictions = 'final', 
                        classProbs = TRUE, 
                        summaryFunction = multiClassSummary,
                        allowParallel = F, 
                        sampling = "up", 
                        returnResamp = "final",
                        selectionFunction = "best"
                       ), 
                      tuneList = list(
                        cforest = caretModelSpec(method = "cforest", 
                                                 tuneGrid = expand.grid(mtry = c(1,2,3))),
                        ranger = caretModelSpec(method = "ranger", 
                                                tuneGrid = expand.grid(mtry = c(1,2,3),
                                                                       splitrule = c("extratrees","gini"),
                                                                       min.node.size = seq(1,100,1))), 
                        LogitBoost = caretModelSpec(method = "LogitBoost", 
                                                    tuneGrid = expand.grid(nIter = seq(1, 200, 1)))
                       ),
                      continue_on_fail=F,
                      metric="AUC")
plot(ensemble$nb)
plot(ensemble$nnet)
plot(ensemble$mlp)

# Fifth Step: Putting all togheter ####

habitats_classification <- function(df){
  nr <- createDataPartition(df[,1], p=0.9, list=FALSE)
  train <- df[nr,]
  test <- df[-nr,]
  # train_IS1: uses 100 clusters centroids ####
  print(paste0("Calculating 100 clusters centroides and building train data..."))
  n_clust <- 100
  rain <- train[train$vegetation.type == "Rainforest",-1]
  rest <- train[train$vegetation.type == "Restinga",-1]
  high <- train[train$vegetation.type == "High.Elevation",-1]
  semi <- train[train$vegetation.type == "Semideciduous",-1]
  rive <- train[train$vegetation.type == "Riverine",-1]
  rock <- train[train$vegetation.type == "Rocky",-1]
  rain <- kmeans(rain, n_clust)
  rest <- kmeans(rest, n_clust)
  high <- kmeans(high, n_clust)
  semi <- kmeans(semi, n_clust)
  rive <- kmeans(rive, n_clust)
  rock <- kmeans(rock, n_clust)
  rain <- rain$centers
  rest <- rest$centers
  high <- high$centers
  semi <- semi$centers
  rive <- rive$centers
  rock <- rock$centers
  vegetation.type <- c(rep("Rainforest",     n_clust), 
                       rep("Semideciduous",  n_clust), 
                       rep("High.Elevation", n_clust), 
                       rep("Restinga",       n_clust), 
                       rep("Riverine",       n_clust), 
                       rep("Rocky",          n_clust))
  families <- as.data.frame(rbind(rain, semi, high, rest, rive, rock))
  train <- data.frame(vegetation.type = as.character(vegetation.type),
                          myrtaceae = as.numeric(families$myrtaceae),
                          fabaceae  = as.numeric(families$fabaceae),
                          rubiaceae = as.numeric(families$rubiaceae))
  
  # Run models ####
  print(paste0("Runing models..."))
  ensemble <- caretList(reformulate(termlabels = colnames(train)[-1], response=colnames(train)[1]),
                        data=train,
                        trControl=trainControl(
                          method = "repeatedcv", 
                          number = 10, 
                          index = createFolds(train[,1], 10, returnTrain = T),
                          repeats = 10, 
                          savePredictions = 'final', 
                          classProbs = TRUE, 
                          summaryFunction = multiClassSummary,
                          allowParallel = F, 
                          sampling = "up", 
                          returnResamp = "final",
                          selectionFunction = "best"
                        ), 
                        tuneList = list(
                          cforest = caretModelSpec(method = "cforest", 
                                                   tuneGrid = expand.grid(mtry = 1)),
                          ranger = caretModelSpec(method = "ranger", 
                                                  tuneGrid = expand.grid(mtry = 1,
                                                                         splitrule = c("gini"),
                                                                         min.node.size = seq(40,100,1))), 
                          LogitBoost = caretModelSpec(method = "LogitBoost", 
                                                      tuneGrid = expand.grid(nIter = seq(50, 100, 1)))
                        ),
                        continue_on_fail=F,
                        metric="AUC")
  
  print(paste0("Calculating AUCs..."))
  values_auc <- NA
  algo <- c("cforest", "ranger", "LogitBoost")
  for (i in 1:length(algo)) {
    v <- ensemble[[i]]$results$AUC[best(ensemble[[i]]$results, "AUC", maximize = T)]
    assign(paste0("auc_",algo[i]), v)
    values_auc[i] <- v
  }
  names(values_auc) <- algo
  
  
  print(paste0("Making predictions..."))
  best_model <- ensemble[[which.max(values_auc)]]
  model_preds <- predict(best_model, newdata=test, type = "prob")
  pred_auc <- multiclass.roc(test$vegetation.type, model_preds)
  
  print(paste0("Done!"))
  return(list(AUCs = pred_auc$auc, 
              Best_Model = best_model,
              Test.Set = test,
              Train.Set = train))
}

n <- 100
set.seed(1)
result <- replicate(n, habitats_classification(df))

saveRDS(result, "result_final.rds")

aucs <- NA
for(i in 1:100){ aucs[i] <- as.numeric(result[,i][1]) }
max(aucs)
min(aucs)
mean(aucs)
sd(aucs)

result_best <- result[,which(aucs == max(aucs))]
saveRDS(result_best, "result_best_final.rds")
result_best$Best_Model











################################################ End of Script ################################################

