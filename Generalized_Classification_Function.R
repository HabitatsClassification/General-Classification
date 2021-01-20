# General Classification Function:
# Open necessary libraries:
library(caretEnsemble)
library(caret)
library(dplyr)
library(here)
library(pROC)
library(dplyr)

# Open NeoTropTree dataframe:
df <- read.csv(here("NTT_habitats_classification.csv"), stringsAsFactors = TRUE)
df <- df[, -1]
head(df)
summary(df)

algo_2nd_step <- c( 
  # SVM
  "lssvmRadial", "svmLinear", "svmLinear2", "svmRadialWeights",
  "svmRadialSigma", "svmRadialCost", "svmRadial", "svmPoly",
  # NeuralNetwork
  "avNNet", "dnn", "mlp", "mlpML", "mlpWeightDecay",
  "mlpWeightDecayML", "monmlp", "multinom", "nnet",
  "pcaNNet", "rbfDDA",
  # Random Forest
  "cforest", "ranger", "Rborist",
  "rf", "RRF", "RRFglobal", "wsrf",
  # Logistic Regression
  "LMT", "LogitBoost", "polr", "vglmAdjCat", "vglmContRatio", "vglmCumulative",
  # Naive Bayes
  "nb", "naive_bayes"
)

n_clust <- 100
n <- 100

# First Step: Framework Selection ####
First_step <- function(df, n) {
  classification <- function(df) {
      habitats <- levels(df[, 1])
      nr <- createDataPartition(df[, 1], p = 0.9, list = FALSE)
      train <- df[nr, ]
      test <- df[-nr, ]
      algo <- c("nb", "rf", "LogitBoost", "nnet", "svmRadial")
      #### Generatind Methods ####
      multiclass <- function(train, test, algo) {
        t <- trainControl(
          method = "repeatedcv",
          number = 10,
          index = createFolds(train[, 1], 10, returnTrain = T),
          repeats = 10,
          savePredictions = "final",
          classProbs = TRUE,
          summaryFunction = multiClassSummary,
          allowParallel = F,
          sampling = "up",
          returnResamp = "final",
          selectionFunction = "best"
        )
        ensemble <- caretList(
          reformulate(termlabels = colnames(train)[-1], response = colnames(train)[1]),
          data = train,
          trControl = t,
          methodList = algo,
          continue_on_fail = T,
          metric = "AUC"
        )
        model_preds <- lapply(ensemble, predict, newdata = test, type = "prob")
        
        values_auc <- NA
        for (i in 1:length(algo)) {
          v <- ensemble[[i]]$results$AUC[best(ensemble[[i]]$results, "AUC", maximize = T)]
          if (v > 0.5) {
            assign(paste0("auc_", algo[i]), v)
            values_auc[i] <- v
          } else {
            assign(paste0("auc_", algo[i]), 0)
            values_auc[i] <- 0
          }
        }
        sum_auc <- sum(values_auc)
        result <- test
        for (i in 1:length(habitats)) {
          df2 <- sapply(model_preds, "[", i)
          df2 <- Map("*", df2, values_auc)
          df2 <- as.data.frame(df2)
          df2 <- rowSums(df2) / sum_auc
          result <- cbind(result, df2)
        }
        colnames(result) <- c(colnames(test), habitats)
        final_prediction <- apply(result[, -(1:ncol(df))], 1, function(x) names(which.max(x)))
        habitat_probs <- NA
        for (i in 1:nrow(result)) {
          habitat_probs[i] <- result[i, colnames(result) == result[, 1][i]]
        }
        result <- cbind(result, final_prediction, habitat_probs)
        
        return(result)
      }
      one_vs_all <- function(train, test, algo) {
        for (i in 1:length(habitats)) {
          train2 <- train
          train2[, 1] <- ifelse(train2[, 1] != habitats[i], "Other", habitats[i]) %>%
            factor(levels = c(habitats[i], "Other"))
          assign(paste0(habitats[i]), train2)
          
          print(paste0(habitats[i]))
        }
        
        habitat_classification <- function(df_habitat, algo) {
          u <- unique(df_habitat[, 1])
          u <- u[u != "Other"]
          t <- trainControl(
            method = "repeatedcv",
            number = 10,
            index = createFolds(df_habitat[, 1], 10, returnTrain = T),
            repeats = 10,
            savePredictions = "final",
            classProbs = TRUE,
            summaryFunction = twoClassSummary,
            allowParallel = F,
            sampling = "down"
          )
          ensemble <- caretList(reformulate(termlabels = colnames(train)[-1], response = colnames(train)[1]),
                                data = df_habitat,
                                trControl = t,
                                methodList = algo,
                                continue_on_fail = T,
                                metric = "ROC",
                                verbose = T
          )
          assign(paste0(u, "_", algo[1]), ensemble)
        }
        
        # Run Function:
        for (i in 1:length(habitats)) {
          assign(paste0(habitats[i], "_result"), habitat_classification(get(habitats[i]), algo))
          print(paste0(habitats[i], " one-vs-all ended!"))
        }
        
        ensembles <- paste0(habitats, "_result")
        
        result <- test
        
        for (j in 1:length(ensembles)) {
          print(j)
          en <- get(ensembles[j])
          
          values_roc <- NA
          for (i in 1:length(algo)) {
            v <- en[[i]]$results$ROC[best(en[[i]]$results, "ROC", maximize = T)]
            if (v > 0.5) {
              assign(paste0("roc_", algo[i]), v)
              values_roc[i] <- v
            } else {
              assign(paste0("roc_", algo[i]), 0)
              values_roc[i] <- 0
            }
          }
          
          sum_roc <- sum(values_roc)
          model_preds <- lapply(en, predict, newdata = test, type = "prob")
          df2 <- sapply(model_preds, "[", habitats[j])
          df2 <- Map("*", df2, values_roc)
          df2 <- as.data.frame(df2)
          df2 <- rowSums(df2) / sum_roc
          result <- cbind(result, df2)
        }
        
        colnames(result) <- c(colnames(test), habitats)
        
        final_prediction <- colnames(result[, -(1:4)])[apply(result[, -(1:4)], 1, which.max)]
        habitat_probs <- NA
        for (i in 1:nrow(result)) {
          habitat_probs[i] <- result[i, colnames(result) == result[, 1][i]]
        }
        result <- cbind(result, final_prediction, habitat_probs)
        return(result)
      }
      result_multiclass <- multiclass(train, test, algo)
      result_one_vs_all <- one_vs_all(train, test, algo)
      #### Comparing Methods ####
      mc.roc <- multiclass.roc(result_multiclass[, 1], result_multiclass$habitat_probs)
      oa.roc <- multiclass.roc(result_one_vs_all[, 1], result_one_vs_all$habitat_probs)
      result <- list("Multiclass" = as.numeric(mc.roc$auc), "One.vs.all" = as.numeric(oa.roc$auc))
      return(result)
    }
  result_framework <- replicate(n, classification(df))
  df_new <- data.frame(
    multiclass = unlist(result_framework[1, ]),
    one_vs_all = unlist(result_framework[2, ])
  )
  df_ttest <- data.frame(
    method = c(rep("multiclass", n), rep("one_vs_all", n)),
    AUC = c(df_new$multiclass, df_new$one_vs_all)
  )
  t_st <- t.test(AUC ~ method, df_ttest)
  mean_multiclass <- mean(df_new$multiclass)
  mean_one_vs_all <- mean(df_new$one_vs_all)
  selected_framework <- ifelse(mean_multiclass > mean_one_vs_all, "multiclass", "one_vs_all")
  return(list(
    DataFrame_TTest = df_ttest,
    mean_multiclass_Fist_step = mean_multiclass,
    mean_one_vs_all_Fist_step = mean_one_vs_all,
    TTest_Fist_step = t_st,
    Selected_Framework = selected_framework
  ))
}
set.seed(1)
s1 <- First_step(df, n)
print(paste0("Selected Framework: ", s1$Selected_Framework, "."))
saveRDS(s1, "s1_GCF.rds")

# Second Step: Selecting Algorithms ####
Second_step <- function(Selected_Framework, algo_2nd_step, n) {
  if (Selected_Framework == "multiclass") {
    paste0("multiclass")
    multiclass <- function(algo_2nd_step, n) {
      resultado <- list()
      for (j in 1:n) {
        print(j)
        nr <- createDataPartition(df[, 1], p = 0.9, list = FALSE)
        train <- df[nr, ]
        test <- df[-nr, ]
        t <- trainControl(
          method = "repeatedcv",
          number = 10,
          index = createFolds(train[, 1], 10, returnTrain = T),
          repeats = 10,
          savePredictions = "final",
          classProbs = TRUE,
          summaryFunction = multiClassSummary,
          allowParallel = T,
          sampling = "up",
          returnResamp = "final",
          selectionFunction = "best"
        )
        ensemble <- caretList(
          reformulate(termlabels = colnames(train)[-1], response = colnames(train)[1]),
          data = train,
          trControl = t,
          methodList = algo_2nd_step,
          continue_on_fail = T,
          metric = "AUC"
        )
        model_preds <- lapply(ensemble, predict, newdata = test, type = "raw")
        x <- lapply(lapply(model_preds, as.numeric), multiclass.roc, response = as.numeric(test$vegetation.type))
        preds_auc <- NA
        for (i in 1:length(names(ensemble))) {
          preds_auc[i] <- as.numeric(x[[i]]$auc)
        }
        resultado[[j]] <- data.frame(
          algo = names(ensemble),
          auc = preds_auc
        )
      }
      return(resultado)
    }
    result_1 <- multiclass(algo_2nd_step, n)
  } else {
    paste0("one_vs_all")
    habitats <- levels(df[, 1])
    one_vs_all <- function(df, habitats, algo_2nd_step, n) {
      resultado <- list()
      for (j in 1:n) {
        print(j)
        nr <- createDataPartition(df[, 1], p = 0.9, list = FALSE)
        train <- df[nr, ]
        test <- df[-nr, ]
        for (i in 1:length(habitats)) {
          train2 <- train
          train2[, 1] <- ifelse(train2[, 1] != habitats[i], "Other", habitats[i]) %>%
            factor(levels = c(habitats[i], "Other"))
          assign(paste0(habitats[i]), train2)
          
          print(paste0(habitats[i]))
        }
        habitat_classification <- function(df_habitat, algo_2nd_step) {
          u <- unique(df_habitat[, 1])
          u <- u[u != "Other"]
          t <- trainControl(
            method = "repeatedcv",
            number = 10,
            index = createFolds(df_habitat[, 1], 10, returnTrain = T),
            repeats = 10,
            savePredictions = "final",
            classProbs = TRUE,
            summaryFunction = twoClassSummary,
            allowParallel = F,
            sampling = "down"
          )
          ensemble <- caretList(reformulate(termlabels = colnames(train)[-1], response = colnames(train)[1]),
                                data = df_habitat,
                                trControl = t,
                                methodList = algo_2nd_step,
                                continue_on_fail = T,
                                metric = "ROC",
                                verbose = T
          )
          assign(paste0(u, "_", algo_2nd_step[1]), ensemble)
        }
        
        # Run Function:
        for (i in 1:length(habitats)) {
          assign(paste0(habitats[i], "_result"), habitat_classification(get(habitats[i]), algo_2nd_step))
          print(paste0(habitats[i], " one-vs-all ended!"))
        }
        
        for (i in 1:length(algo_2nd_step)) {
          df2 <- test
          for (j in 1:length(ensembles)) {
            x <- get(ensembles[j])[[i]]
            model_preds <- predict(x, newdata = test, type = "prob")
            df2 <- cbind(df2, model_preds[, 1])
          }
          colnames(df2) <- c(colnames(test), habitats)
          final_prediction <- as.factor(colnames(df2[, -(1:ncol(test))])[apply(df2[, -(1:ncol(test))], 1, which.max)])
          x <- pROC::multiclass.roc(response = as.numeric(test$vegetation.type), predictor = as.numeric(final_prediction))
          result[i, 1] <- algo_2nd_step[i]
          result[i, 2] <- as.numeric(x$auc)
        }
        
        return(result)
      }
    }
    result_1 <- one_vs_all(df, habitats, algo_2nd_step, n)
  }
  
  result_df <- bind_rows(result_1)
  bmean <- aggregate(result_df$auc, by = list(Category = result_df$algo), FUN = mean)
  bmax <- aggregate(result_df$auc, by = list(Category = result_df$algo), FUN = max)
  bsum <- bmean$x + bmax$x
  result_aov <- aov(auc ~ algo, result_df)
  summary(result_aov)
  tuk <- TukeyHSD(result_aov)
  tuk <- as.data.frame(tuk$algo)
  result_df2 <- data.frame(
    Algorithm = bmean[, 1],
    Mean = bmean[, 2],
    Max = bmax[, 2],
    Sum = bsum
  )
  result_df2 <- result_df2[order(result_df2$Sum, decreasing = T), ]
  selected_algos <- as.character(result_df2[1:3, 1])
  return(list(
    Result_Second_step = result_df2,
    Algo_selected_Second_step = selected_algos,
    AOV_Second_step = result_aov,
    TUK_Second_step = tuk
  ))
}
set.seed(1)
s2 <- Second_step(s1$Selected_Framework, algo_2nd_step, n)
print(paste0("Selected Algorithms: ", s2$Result_Second_step$Algorithm[1], ", ",
             s2$Result_Second_step$Algorithm[2], ", ",
             s2$Result_Second_step$Algorithm[3], "."))
saveRDS(s2,"s2_GCF.rds")

# Third Step: Instance Selection ####
Third_step <- function(df, Selected_Framework, algo, n_clust, n_clust2, n) {
  instance_selection <- function(df, algo, n_clust, n_clust2) {
    nr <- createDataPartition(df[, 1], p = 0.9, list = FALSE)
    train <- df[nr, ]
    test <- df[-nr, ]
    habitats <- levels(train[, 1])
    
    # train_IS0: no instance selection ####
    print(paste0("Train IS0 Started."))
    train_IS0 <- train
    
    # train_IS1: uses 100 clusters centroides ####
    print(paste0("Train IS1 Started."))
    for (i in 1:length(habitats)) {
      x <- train[train[, 1] == habitats[i], -1]
      x <- kmeans(x, n_clust)
      v <- data.frame(
        vegetation.type = as.factor(rep(paste(habitats[i]), n_clust)),
        x$centers
      )
      if (i == 1) {
        train_IS1 <- v
      } else {
        train_IS1 <- rbind(train_IS1, v)
      }
    }
    
    # train_IS2: removes outliers from 50 clusters, scales up ####
    print(paste0("Train IS2 Started."))
    for (j in 1:length(habitats)) {
      df2 <- train[train[, 1] == habitats[j], -1]
      df3 <- kmeans(df2, n_clust)
      for (i in 1:n_clust) {
        df_clust <- df2[df3$cluster == i, -4]
        x <- sort(as.matrix(dist(rbind(df3$centers[i, ], df_clust)))[, 1])[-1]
        if ((length(x) %% 2) == 0) {
          num <- length(x) / 2
        } else {
          num <- length(x) / 2 + 1
        }
        x <- x[1:num]
        df_clust <- df[names(x), ]
        if (i == 1) {
          assign(paste0("train_", habitats[j]), df_clust)
        } else {
          assign(paste0("train_", habitats[j]), rbind(get(paste0("train_", habitats[j])), df_clust))
        }
      }
      if (j == 1) {
        train_IS2 <- get(paste0("train_", habitats[j]))
      } else {
        train_IS2 <- rbind(train_IS2, get(paste0("train_", habitats[j])))
      }
    }
    
    # train_IS3: removes outliers from one cluster, scales up ####
    print(paste0("Train IS3 Started."))
    for (j in 1:length(habitats)) {
      df2 <- train[train[, 1] == habitats[j], -1]
      df3 <- kmeans(df2, 1)
      for (i in 1:1) {
        df_clust <- df2[df3$cluster == i, -4]
        x <- sort(as.matrix(dist(rbind(df3$centers[i, ], df_clust)))[, 1])[-1]
        if ((length(x) %% 2) == 0) {
          num <- length(x) / 2
        } else {
          num <- length(x) / 2 + 1
        }
        x <- x[1:num]
        df_clust <- df[names(x), ]
        if (i == 1) {
          assign(paste0("train_", habitats[j]), df_clust)
        } else {
          assign(paste0("train_", habitats[j]), rbind(get(paste0("train_", habitats[j])), df_clust))
        }
      }
      if (j == 1) {
        train_IS3 <- get(paste0("train_", habitats[j]))
      } else {
        train_IS3 <- rbind(train_IS3, get(paste0("train_", habitats[j])))
      }
    }
    
    # train_IS4: removes outliers from one cluster and make 50 clusters out of the result ####
    print(paste0("Train IS4 Started."))
    for (j in 1:length(habitats)) {
      df2 <- train[train[, 1] == habitats[j], -1]
      df3 <- kmeans(df2, 1)
      for (i in 1:1) {
        df_clust <- df2[df3$cluster == i, -4]
        x <- sort(as.matrix(dist(rbind(df3$centers[i, ], df_clust)))[, 1])[-1]
        if ((length(x) %% 2) == 0) {
          num <- length(x) / 2
        } else {
          num <- length(x) / 2 + 1
        }
        x <- x[1:num]
        df_clust <- df[names(x), ]
        if (i == 1) {
          assign(paste0("train_", habitats[j]), df_clust)
        } else {
          assign(paste0("train_", habitats[j]), rbind(get(paste0("train_", habitats[j])), df_clust))
        }
      }
      if (j == 1) {
        train_IS4.1 <- get(paste0("train_", habitats[j]))
      } else {
        train_IS4.1 <- rbind(train_IS4.1, get(paste0("train_", habitats[j])))
      }
    }
    
    # build 50 clusters:
    for (i in 1:length(habitats)) {
      x <- train_IS4.1[train_IS4.1[, 1] == habitats[i], -1]
      x <- kmeans(x, n_clust2)
      v <- data.frame(
        vegetation.type = as.factor(rep(paste(habitats[i]), n_clust2)),
        x$centers
      )
      if (i == 1) {
        train_IS4 <- v
      } else {
        train_IS4 <- rbind(train_IS4, v)
      }
      print(i)
    }
    
    train_data <- list(train_IS0, train_IS1, train_IS2, train_IS3, train_IS4)
    # Finnished Train Data Building ####
    # Apply Function ####
    if (Selected_Framework == "multiclass") {
      paste0("multiclass")
      multiclass <- function(tr) {
        habitats <- levels(df[, 1])
        t <- trainControl(
          method = "repeatedcv",
          number = 10,
          index = createFolds(tr[, 1], 10, returnTrain = T),
          repeats = 10,
          savePredictions = "final",
          classProbs = TRUE,
          summaryFunction = multiClassSummary,
          allowParallel = F,
          sampling = "up",
          returnResamp = "final",
          selectionFunction = "best"
        )
        ensemble <- caretList(
          reformulate(termlabels = colnames(tr)[-1], response = colnames(tr)[1]),
          data = tr,
          trControl = t,
          methodList = algo,
          continue_on_fail = T,
          metric = "AUC"
        )
        model_preds <- lapply(ensemble, predict, newdata = test, type = "prob")
        
        values_auc <- NA
        for (i in 1:length(algo)) {
          v <- ensemble[[i]]$results$AUC[best(ensemble[[i]]$results, "AUC", maximize = T)]
          if (v > 0.5) {
            assign(paste0("auc_", algo[i]), v)
            values_auc[i] <- v
          } else {
            assign(paste0("auc_", algo[i]), 0)
            values_auc[i] <- 0
          }
        }
        sum_auc <- sum(values_auc)
        result <- test
        for (i in 1:length(habitats)) {
          df2 <- sapply(model_preds, "[", i)
          df2 <- Map("*", df2, values_auc)
          df2 <- as.data.frame(df2)
          df2 <- rowSums(df2) / sum_auc
          result <- cbind(result, df2)
        }
        colnames(result) <- c(colnames(test), habitats)
        final_prediction <- data.frame(final_prediction = as.factor(apply(result[, -(1:ncol(df))], 1, function(x) names(which.max(x)))))
        x <- multiclass.roc(response = as.numeric(final_prediction[, 1]), predictor = as.numeric(test[, 1]))
        x <- as.numeric(x$auc)
        return(x)
      }
      print(paste0("Applying data to multiclass function"))
      z <- mapply(multiclass, train_data)
      z2 <- data.frame(IS = c("IS0", "IS1", "IS2", "IS3", "IS4"), z)
      print(paste0("Instance Selection Finnished"))
      return(z)
    } else {
      paste0("one_vs_all")
      one_vs_all <- function(train) {
        for (i in 1:length(habitats)) {
          train2 <- train
          train2[, 1] <- ifelse(train2[, 1] != habitats[i], "Other", habitats[i]) %>%
            factor(levels = c(habitats[i], "Other"))
          assign(paste0(habitats[i]), train2)
          
          print(paste0(habitats[i]))
        }
        
        habitat_classification <- function(df_habitat, algo) {
          u <- unique(df_habitat[, 1])
          u <- u[u != "Other"]
          t <- trainControl(
            method = "repeatedcv",
            number = 10,
            index = createFolds(df_habitat[, 1], 10, returnTrain = T),
            repeats = 10,
            savePredictions = "final",
            classProbs = TRUE,
            summaryFunction = twoClassSummary,
            allowParallel = F,
            sampling = "down"
          )
          ensemble <- caretList(reformulate(termlabels = colnames(train)[-1], response = colnames(train)[1]),
                                data = df_habitat,
                                trControl = t,
                                methodList = algo,
                                continue_on_fail = T,
                                metric = "ROC",
                                verbose = T
          )
          assign(paste0(u, "_", algo[1]), ensemble)
        }
        
        # Run Function:
        for (i in 1:length(habitats)) {
          assign(paste0(habitats[i], "_result"), habitat_classification(get(habitats[i]), algo))
          print(paste0(habitats[i], " one-vs-all ended!"))
        }
        
        ensembles <- paste0(habitats, "_result")
        
        result <- test
        
        for (j in 1:length(ensembles)) {
          print(j)
          en <- get(ensembles[j])
          
          values_roc <- NA
          for (i in 1:length(algo)) {
            v <- en[[i]]$results$ROC[best(en[[i]]$results, "ROC", maximize = T)]
            if (v > 0.5) {
              assign(paste0("roc_", algo[i]), v)
              values_roc[i] <- v
            } else {
              assign(paste0("roc_", algo[i]), 0)
              values_roc[i] <- 0
            }
          }
          
          sum_roc <- sum(values_roc)
          model_preds <- lapply(en, predict, newdata = test, type = "prob")
          df2 <- sapply(model_preds, "[", habitats[j])
          df2 <- Map("*", df2, values_roc)
          df2 <- as.data.frame(df2)
          df2 <- rowSums(df2) / sum_roc
          result <- cbind(result, df2)
        }
        
        colnames(result) <- c(colnames(test), habitats)
        
        final_prediction <- as.factor(colnames(result[, -(1:4)])[apply(result[, -(1:4)], 1, which.max)])
        habitat_probs <- NA
        for (i in 1:nrow(result)) {
          habitat_probs[i] <- result[i, colnames(result) == result[, 1][i]]
        }
        result <- cbind(result, final_prediction, habitat_probs)
        oa.roc <- multiclass.roc(as.numeric(test[, 1]), as.numeric(final_prediction))
        x <- as.numeric(oa.roc$auc)
        return(x)
      }
      print(paste0("Applying data to one_vs_all function"))
      z <- mapply(one_vs_all, train_data)
      z2 <- data.frame(IS = c("IS0", "IS1", "IS2", "IS3", "IS4"), z)
      print(paste0("Instance Selection Finnished"))
      return(z)
    }
  }
  result_instance_selection <- replicate(n, instance_selection(df, algo, n_clust = 100, n_clust2 = 50))
  result_instance_selection <- as.data.frame(result_instance_selection)
  rownames(result_instance_selection) <- c("IS0", "IS1", "IS2", "IS3", "IS4")
  df_aov <- data.frame(
    method = c(
      rep("IS0", n),
      rep("IS1", n),
      rep("IS2", n),
      rep("IS3", n),
      rep("IS4", n)
    ),
    AUC = c(
      as.numeric(result_instance_selection[1, ]),
      as.numeric(result_instance_selection[2, ]),
      as.numeric(result_instance_selection[3, ]),
      as.numeric(result_instance_selection[4, ]),
      as.numeric(result_instance_selection[5, ])
    )
  )
  
  bmean <- aggregate(AUC ~ method, df_aov, FUN = mean)[, 2]
  bmax <- aggregate(AUC ~ method, df_aov, FUN = max)[, 2]
  bsum <- bmean + bmax
  Result_Third_step <- data.frame(
    IS = c("IS0", "IS1", "IS2", "IS3", "IS4"),
    Mean = bmean,
    Max = bmax,
    Sum = bsum
  )
  Result_Third_step <- Result_Third_step[order(Result_Third_step$Sum, decreasing = T), ]
  
  result_aov <- aov(AUC ~ method, df_aov)
  summary(result_aov)
  summary.lm(result_aov)
  tuk <- TukeyHSD(result_aov)
  
  auc_plot <- ggplot(df_aov, aes(x = method, y = AUC, fill = method)) +
    geom_boxplot() +
    labs(x = "Method") +
    theme(legend.position = "none")
  return(list(
    Result_Third_step = Result_Third_step,
    AOV = result_aov,
    AOV_summary = summary(result_aov),
    AOV_summary.lm = summary.lm(result_aov),
    Tukey = tuk <- TukeyHSD(result_aov),
    AUC_plot = auc_plot
  ))
}
set.seed(1)
s3 <- Third_step(df,Selected_Framework = s1$Selected_Framework, algo = s2$Algo_selected_Second_step, n_clust, n_clust2 = n_clust/2, n)
print(paste0("Instance Selection Selected: ", s3$Result_Third_step$IS[1], "."))
saveRDS(s3,"s3_GCF.rds")

# Fourth Step: Algorithms Tuning ####
set.seed(1)
nr <- createDataPartition(df[, 1], p = 0.9, list = FALSE)
train <- df[nr, ]
test <- df[-nr, ]
ensemble <- caretList(reformulate(termlabels = colnames(train)[-1], response = colnames(train)[1]),
                      data = train,
                      trControl = trainControl(
                        method = "repeatedcv",
                        number = 10,
                        index = createFolds(train[, 1], 10, returnTrain = T),
                        repeats = 10,
                        savePredictions = "final",
                        classProbs = TRUE,
                        summaryFunction = multiClassSummary,
                        allowParallel = F,
                        sampling = "up",
                        returnResamp = "final",
                        selectionFunction = "best"
                      ),
                      tuneList = list(
                        cforest = caretModelSpec(
                          method = "cforest",
                          tuneGrid = expand.grid(mtry = c(1, 2, 3))
                        ),
                        ranger = caretModelSpec(
                          method = "ranger",
                          tuneGrid = expand.grid(
                            mtry = c(1, 2, 3),
                            splitrule = c("extratrees", "gini"),
                            min.node.size = seq(1, 100, 1)
                          )
                        ),
                        LogitBoost = caretModelSpec(
                          method = "LogitBoost",
                          tuneGrid = expand.grid(nIter = seq(1, 200, 1))
                        )
                      ),
                      continue_on_fail = F,
                      metric = "AUC"
)
plot(ensemble$cforest)
plot(ensemble$ranger)
plot(ensemble$LogitBoost)

# Fifth Step: Putting results togheter ####
habitats_classification <- function(df) {
  ##### Data Building: ####
  nr <- createDataPartition(df[, 1], p = 0.9, list = FALSE)
  train <- df[nr, ]
  test <- df[-nr, ]
  # train_IS1: uses 100 clusters centroides ####
  if(framework_result$Third_step$Result_Third_step$IS[1] == "IS1"){
    print(paste0("Train IS1 Started."))
    for (i in 1:length(habitats)) {
      x <- train[train[, 1] == habitats[i], -1]
      x <- kmeans(x, n_clust)
      v <- data.frame(
        vegetation.type = as.factor(rep(paste(habitats[i]), n_clust)),
        x$centers
      )
      if (i == 1) {
        train_IS1 <- v
      } else {
        train_IS1 <- rbind(train_IS1, v)
      }
    }
    train <- train_IS1
  }

  ##### Run models ####
  print(paste0("Runing multiclass models..."))
  ensemble <- caretList(reformulate(termlabels = colnames(train)[-1], response = colnames(train)[1]),
                        data = train,
                        trControl = trainControl(
                          method = "repeatedcv",
                          number = 10,
                          index = createFolds(train[, 1], 10, returnTrain = T),
                          repeats = 10,
                          savePredictions = "final",
                          classProbs = TRUE,
                          summaryFunction = multiClassSummary,
                          allowParallel = F,
                          sampling = "up",
                          returnResamp = "final",
                          selectionFunction = "best"
                        ),
                        tuneList = list(
                          cforest = caretModelSpec(
                            method = "cforest",
                            tuneGrid = expand.grid(mtry = 1)
                          ),
                          ranger = caretModelSpec(
                            method = "ranger",
                            tuneGrid = expand.grid(
                              mtry = 1,
                              splitrule = c("gini"),
                              min.node.size = seq(40, 100, 1)
                            )
                          ),
                          LogitBoost = caretModelSpec(
                            method = "LogitBoost",
                            tuneGrid = expand.grid(nIter = seq(50, 100, 1))
                          )
                        ),
                        continue_on_fail = F,
                        metric = "AUC"
  )
  
  print(paste0("Calculating AUCs..."))
  values_auc <- NA
  algo <- c("cforest", "ranger", "LogitBoost")
  for (i in 1:length(algo)) {
    v <- ensemble[[i]]$results$AUC[best(ensemble[[i]]$results, "AUC", maximize = T)]
    assign(paste0("auc_", algo[i]), v)
    values_auc[i] <- v
  }
  names(values_auc) <- algo
  
  print(paste0("Making predictions..."))
  best_model <- ensemble[[which.max(values_auc)]]
  model_preds <- predict(best_model, newdata = test, type = "prob")
  pred_auc <- multiclass.roc(test$vegetation.type, model_preds)
  
  print(paste0("Done!"))
  return(list(
    AUCs = pred_auc$auc,
    Best_Model = best_model,
    Test.Set = test,
    Train.Set = train
  ))
}
set.seed(1)
result <- replicate(n, habitats_classification(df))
saveRDS(result, "result_final.rds")

# Extracting Best Model:
aucs <- NA
for (i in 1:100) {
  aucs[i] <- as.numeric(result[, i][1])
}
result_best <- result[, which(aucs == max(aucs))]
saveRDS(result_best, "result_best_final.rds")
result_best$Best_Model


################################################ End of Script ################################################
