library(caretEnsemble)
library(caret)
library(dplyr)
library(here)
library(pROC)
library(doParallel)
cl <- makePSOCKcluster(3)
registerDoParallel(cl)

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

habitats_training_1 <- function(n){
  models <- NA
  final_aucs <- NA
  all_aucs <- list()
  all_models <- list()
  all_best_tunes <- list()
  train_sets <- list()
  test_sets <- list()
  n_clust <- 50
  for (j in 1:n){
    # Clustering
    nr <- createDataPartition(df[,1], p=0.9, list=FALSE)
    train <- df[nr,]
    test <- df[-nr,]
    #train = 50 clusters
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
    train_sets[[j]] <- train
    test_sets[[j]] <- test
    # apply function 
    ensemble <- caretList(reformulate(termlabels = colnames(train)[-1], response=colnames(train)[1]),
                          data=train,
                          trControl=trainControl(
                            method = "repeatedcv", 
                            number = 10, 
                            index = createFolds(train[,1], 10, returnTrain = T),
                            repeats = 10, 
                            savePredictions = 'final', 
                            classProbs = T, 
                            summaryFunction = multiClassSummary,
                            allowParallel = F, 
                            returnResamp = "final",
                            selectionFunction = "best"), 
                          tuneList = list(
                            naive_bayes = caretModelSpec(method = "naive_bayes", 
                                                         tuneGrid = expand.grid(laplace = c(0),
                                                                                usekernel = TRUE,
                                                                                adjust = seq(0.65, 0.75, 0.01))),
                            nnet = caretModelSpec(method = "nnet", 
                                                  tuneGrid = expand.grid(size = seq(10, 15, 1), # 10, 30
                                                                         decay = seq(0.6,0.8,0.1))), # 0.65, 0.8, 0.05
                            mlp = caretModelSpec(method = "mlp", 
                                                 tuneGrid = expand.grid(size = seq(10, 20, 1)))),
                          continue_on_fail=F,
                          metric="AUC")
    model_preds <- lapply(ensemble, predict.train, newdata=test, type="raw")
    x <- lapply(lapply(model_preds, as.numeric), multiclass.roc, response = as.numeric(test$vegetation.type))
    preds_auc <- NA
    for (i in 1:length(names(ensemble))){
      preds_auc[i] <- as.numeric(x[[i]]$auc)
    }
    models[j] <- ensemble[which.max(preds_auc)]
    final_aucs[j] <- preds_auc[which.max(preds_auc)]
    all_aucs[[j]] <- preds_auc
    all_models[[j]] <- ensemble
    best_tunes <- list()
    for (k in 1:length(names(ensemble))){
      best_tunes[[k]] <- ensemble[[k]]$bestTune
    }
    all_best_tunes[[j]] <- best_tunes
  } 
  best_model <- models[which.max(final_aucs)]
  
  return(list(Models = models, 
              AUCs = final_aucs, 
              Best_Model = best_model, 
              All_Best_Tunes = all_best_tunes,
              All_Models = all_models,
              All_AUCs = all_aucs,
              Train_sets = train_sets, 
              Test_sets = test_sets))
} # uses 50 clusters centroides

habitats_training_2 <- function(n){
  models <- NA
  final_aucs <- NA
  all_aucs <- list()
  all_models <- list()
  all_best_tunes <- list()
  train_sets <- list()
  test_sets <- list()
  n_clust <- 50
  for (j in 1:n){
    # Clustering
    nr <- createDataPartition(df[,1], p=0.9, list=FALSE)
    train <- df[nr,]
    test <- df[-nr,]
    #train = 50 clusters
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
    train <- rbind(train_rain, train_rest, train_high, train_semi, train_rive, train_rock)
 
    train_sets[[j]] <- train
    test_sets[[j]] <- test
    # apply function 
    ensemble <- caretList(reformulate(termlabels = colnames(train)[-1], response=colnames(train)[1]),
                          data=train,
                          trControl=trainControl(
                            method = "repeatedcv", 
                            number = 10, 
                            index = createFolds(train[,1], 10, returnTrain = T),
                            repeats = 10, 
                            savePredictions = 'final', 
                            classProbs = T, 
                            summaryFunction = multiClassSummary,
                            allowParallel = F, 
                            returnResamp = "final",
                            selectionFunction = "best"), 
                          tuneList = list(
                            naive_bayes = caretModelSpec(method = "naive_bayes", 
                                                         tuneGrid = expand.grid(laplace = c(0),
                                                                                usekernel = TRUE,
                                                                                adjust = seq(0.65, 0.75, 0.01))),
                            nnet = caretModelSpec(method = "nnet", 
                                                  tuneGrid = expand.grid(size = seq(10, 15, 1), # 10, 30
                                                                         decay = seq(0.6,0.8,0.1))), # 0.65, 0.8, 0.05
                            mlp = caretModelSpec(method = "mlp", 
                                                 tuneGrid = expand.grid(size = seq(10, 20, 1)))),
                          continue_on_fail=F,
                          metric="AUC")
    model_preds <- lapply(ensemble, predict.train, newdata=test, type="raw")
    x <- lapply(lapply(model_preds, as.numeric), multiclass.roc, response = as.numeric(test$vegetation.type))
    preds_auc <- NA
    for (i in 1:length(names(ensemble))){
      preds_auc[i] <- as.numeric(x[[i]]$auc)
    }
    models[j] <- ensemble[which.max(preds_auc)]
    final_aucs[j] <- preds_auc[which.max(preds_auc)]
    all_aucs[[j]] <- preds_auc
    all_models[[j]] <- ensemble
    best_tunes <- list()
    for (k in 1:length(names(ensemble))){
      best_tunes[[k]] <- ensemble[[k]]$bestTune
    }
    all_best_tunes[[j]] <- best_tunes
  } 
  best_model <- models[which.max(final_aucs)]
  
  return(list(Models = models, 
              AUCs = final_aucs, 
              Best_Model = best_model, 
              All_Best_Tunes = all_best_tunes,
              All_Models = all_models,
              All_AUCs = all_aucs,
              Train_sets = train_sets, 
              Test_sets = test_sets))
} # removes outliers from 50 clusters, scales up

habitats_training_3 <- function(n){
  models <- NA
  final_aucs <- NA
  all_aucs <- list()
  all_models <- list()
  all_best_tunes <- list()
  train_sets <- list()
  test_sets <- list()
  n_clust <- 1
  for (j in 1:n){
    # Clustering
    nr <- createDataPartition(df[,1], p=0.9, list=FALSE)
    train <- df[nr,]
    test <- df[-nr,]
    #train = 1 clusters
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
    train <- rbind(train_rain, train_rest, train_high, train_semi, train_rive, train_rock)
    
    train_sets[[j]] <- train
    test_sets[[j]] <- test
    # apply function 
    ensemble <- caretList(reformulate(termlabels = colnames(train)[-1], response=colnames(train)[1]),
                          data=train,
                          trControl=trainControl(
                            method = "repeatedcv", 
                            number = 10, 
                            index = createFolds(train[,1], 10, returnTrain = T),
                            repeats = 10, 
                            savePredictions = 'final', 
                            classProbs = T, 
                            summaryFunction = multiClassSummary,
                            allowParallel = F, 
                            sampling = "up",
                            returnResamp = "final",
                            selectionFunction = "best"), 
                          tuneList = list(
                            naive_bayes = caretModelSpec(method = "naive_bayes", 
                                                         tuneGrid = expand.grid(laplace = c(0),
                                                                                usekernel = TRUE,
                                                                                adjust = seq(0.65, 0.75, 0.01))),
                            nnet = caretModelSpec(method = "nnet", 
                                                  tuneGrid = expand.grid(size = seq(10, 15, 1), # 10, 30
                                                                         decay = seq(0.6,0.8,0.1))), # 0.65, 0.8, 0.05
                            mlp = caretModelSpec(method = "mlp", 
                                                 tuneGrid = expand.grid(size = seq(10, 20, 1)))),
                          continue_on_fail=F,
                          metric="AUC")
    model_preds <- lapply(ensemble, predict.train, newdata=test, type="raw")
    x <- lapply(lapply(model_preds, as.numeric), multiclass.roc, response = as.numeric(test$vegetation.type))
    preds_auc <- NA
    for (i in 1:length(names(ensemble))){
      preds_auc[i] <- as.numeric(x[[i]]$auc)
    }
    models[j] <- ensemble[which.max(preds_auc)]
    final_aucs[j] <- preds_auc[which.max(preds_auc)]
    all_aucs[[j]] <- preds_auc
    all_models[[j]] <- ensemble
    best_tunes <- list()
    for (k in 1:length(names(ensemble))){
      best_tunes[[k]] <- ensemble[[k]]$bestTune
    }
    all_best_tunes[[j]] <- best_tunes
  } 
  best_model <- models[which.max(final_aucs)]
  
  return(list(Models = models, 
              AUCs = final_aucs, 
              Best_Model = best_model, 
              All_Best_Tunes = all_best_tunes,
              All_Models = all_models,
              All_AUCs = all_aucs,
              Train_sets = train_sets, 
              Test_sets = test_sets))
} # removes outliers from one cluster, scales up

habitats_training_4 <- function(n){
  models <- NA
  final_aucs <- NA
  all_aucs <- list()
  all_models <- list()
  all_best_tunes <- list()
  train_sets <- list()
  test_sets <- list()
 
  for (j in 1:n){
    # Clustering
    nr <- createDataPartition(df[,1], p=0.9, list=FALSE)
    train <- df[nr,]
    test <- df[-nr,]
    #train = 1 cluster
    n_clust <- 1
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
    train <- rbind(train_rain, train_rest, train_high, train_semi, train_rive, train_rock)
    
    # build n clusters:
    n_clust <- 30
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
    
    train_sets[[j]] <- train
    test_sets[[j]] <- test
    # apply function 
    ensemble <- caretList(reformulate(termlabels = colnames(train)[-1], response=colnames(train)[1]),
                          data=train,
                          trControl=trainControl(
                            method = "repeatedcv", 
                            number = 10, 
                            index = createFolds(train[,1], 10, returnTrain = T),
                            repeats = 10, 
                            savePredictions = 'final', 
                            classProbs = T, 
                            summaryFunction = multiClassSummary,
                            allowParallel = F, 
                            returnResamp = "final",
                            selectionFunction = "best"), 
                          tuneList = list(
                            naive_bayes = caretModelSpec(method = "naive_bayes", 
                                                         tuneGrid = expand.grid(laplace = c(0),
                                                                                usekernel = TRUE,
                                                                                adjust = seq(0.65, 0.75, 0.01))),
                            nnet = caretModelSpec(method = "nnet", 
                                                  tuneGrid = expand.grid(size = seq(10, 15, 1), # 10, 30
                                                                         decay = seq(0.6,0.8,0.1))), # 0.65, 0.8, 0.05
                            mlp = caretModelSpec(method = "mlp", 
                                                 tuneGrid = expand.grid(size = seq(10, 20, 1)))),
                          continue_on_fail=F,
                          metric="AUC")
    model_preds <- lapply(ensemble, predict.train, newdata=test, type="raw")
    x <- lapply(lapply(model_preds, as.numeric), multiclass.roc, response = as.numeric(test$vegetation.type))
    preds_auc <- NA
    for (i in 1:length(names(ensemble))){
      preds_auc[i] <- as.numeric(x[[i]]$auc)
    }
    models[j] <- ensemble[which.max(preds_auc)]
    final_aucs[j] <- preds_auc[which.max(preds_auc)]
    all_aucs[[j]] <- preds_auc
    all_models[[j]] <- ensemble
    best_tunes <- list()
    for (k in 1:length(names(ensemble))){
      best_tunes[[k]] <- ensemble[[k]]$bestTune
    }
    all_best_tunes[[j]] <- best_tunes
  } 
  best_model <- models[which.max(final_aucs)]
  
  return(list(Models = models, 
              AUCs = final_aucs, 
              Best_Model = best_model, 
              All_Best_Tunes = all_best_tunes,
              All_Models = all_models,
              All_AUCs = all_aucs,
              Train_sets = train_sets, 
              Test_sets = test_sets))
} # removes outliers from one cluster and make 30 clusters out of the result

test_1 <- habitats_training_1(4)
test_2 <- habitats_training_2(4)
test_3 <- habitats_training_3(4)
test_4 <- habitats_training_4(2)


# visualyze clusters:
# library("factoextra")
#fviz_cluster(rain2, data = rain1,
#             geom = "point",
#             ellipse.type = "convex",
#             show.clust.cent = T,
#             outlier.color = "black",
#             ggtheme = theme_minimal())

#PCA test:
#library("factoextra")
#p <- prcomp(train[train$vegetation.type=="Rainforest",-1])
#fviz_pca_ind(p, geom.ind = "point", pointshape = 21, 
#             pointsize = 2, 
#             fill.ind = train[train$vegetation.type=="Rainforest",1], 
#             col.ind = "black", 
#             palette = "jco", 
#             addEllipses = TRUE,
#             label = "var",
#             col.var = "black",
#             repel = TRUE,
#             legend.title = "Diagnosis") +
#  ggtitle("2D PCA-plot from 30 feature dataset") +
#  theme(plot.title = element_text(hjust = 0.5))