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

## to normalize data: (não demonstrou resultados muito melhores)
#df2 <- t(apply(df[-1], 1, prop.table))
#df <- cbind(df, df2)
#df <- df[,-(2:4)]

######### Clustering
nr <- createDataPartition(df[,1], p=0.9, list=FALSE)
train <- df[nr,]
test <- df[-nr,]

#train = clusters
table(train$vegetation.type)

rain1 <- train[train$vegetation.type == "Rainforest",-1]
rest1 <- train[train$vegetation.type == "Restinga",-1]
high1 <- train[train$vegetation.type == "High.Elevation",-1]
semi1 <- train[train$vegetation.type == "Semideciduous",-1]
rive1 <- train[train$vegetation.type == "Riverine",-1]

rain2 <- kmeans(rain1, 65)
rest2 <- kmeans(rest1, 65)
high2 <- kmeans(high1, 65)
semi2 <- kmeans(semi1, 65)
rive2 <- kmeans(rive1, 65)

rain3 <- rain2$centers
rest3 <- rest2$centers
high3 <- high2$centers
semi3 <- semi2$centers
rive3 <- rive2$centers

vegetation.type <- c(rep("Rainforest", 65), 
                     rep("Semideciduous", 65), 
                     rep("High.Elevation", 65), 
                     rep("Restinga", 65), 
                     rep("Riverine", 65), 
                     rep("Rocky", 65))
train <- cbind(vegetation.type, rbind(rain3, semi3, high3, rest3, rive3, train[train$vegetation.type == "Rocky",-1]))
habitats <- levels(train[,1])
table(train$vegetation.type)
table(test$vegetation.type)
######### 

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

habitats_training <- function(n){
  models <- NA
  final_aucs <- NA
  for (j in 1:n){
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
                            preProcOptions=list(k=65),
                            allowParallel = T, 
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
                          continue_on_fail=T,
                          metric="AUC")
    model_preds <- lapply(ensemble, predict, newdata=test, type="raw")
    x <- lapply(lapply(model_preds, as.numeric), multiclass.roc, response = as.numeric(test$vegetation.type))
    preds_auc <- NA
    for (i in 1:length(names(ensemble))){
      preds_auc[i] <- as.numeric(x[[i]]$auc)
    }
    models[j] <- ensemble[which.max(preds_auc)]
    final_aucs[j] <- preds_auc[which.max(preds_auc)]
  } 
  best_model <- models[which.max(final_aucs)]
  return(list(Models = models, AUCs = final_aucs, Best_model = best_model))
}
  
  
test_1 <- habitats_training(1)
test_2 <- habitats_training(2)
test_3 <- habitats_training(3)
test_4 <- habitats_training(4)


  
  
  
  
  
  