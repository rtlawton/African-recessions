if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(readr)) install.packages("readr", repos = "http://cran.us.r-project.org")
if(!require(corrr)) install.packages("corrr", repos = "http://cran.us.r-project.org")
if(!require(MLmetrics)) install.packages("MLmetrics", repos = "http://cran.us.r-project.org")
if(!require(randomForest)) install.packages("randomForest", repos = "http://cran.us.r-project.org")
if(!require(kableExtra)) installed.packages("kableExtra", repos = "http://cran.us.r-project.org")

# read data from git repository

africa_recession <- read_csv(
  "https://raw.githubusercontent.com/rtlawton/African-recessions/master/africa_recession.csv",
  col_types = cols(growthbucket = col_integer()))
variable_def <- read_csv(
  "https://raw.githubusercontent.com/rtlawton/African-recessions/master/VariableDefinitions.csv")

# check for missing values

sapply(africa_recession, function(x) sum(is.na(x))) %>% view()

# convert growthbucket to factor y, make "Recession" the first (positive) level
y <- factor(ifelse(africa_recession$growthbucket == 0, "growth", 
                   "recession"))
y <- factor(y, levels = c("recession", "growth"))
#Replace growthbucket by new factor column y
africa_recession<- africa_recession %>% 
  mutate(y = y) %>%
  select(-growthbucket)

# show class imbalance
imbalance <- round(mean(y == "recession"),4)
imbalance

# Look at data distributions for columns
dist_plot <- africa_recession%>%select(1:12)%>%
  gather()%>%
  ggplot(aes(value))+
  facet_wrap(~ key, nrow = 3, ncol = 4, scales = "free") +
  geom_density() + 
  theme(axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.text.y=element_blank(),
        axis.ticks.y=element_blank())
dist_plot

# Deal with skewed distributions by applying a log10 transformation and renaming
#
# First identify which variables have skewed distributions
# Criterion is min < 4sds below mean XOR max > 4sds above mean
# Also eliminate variables with negative values (eg variables representing a change)
# as these will not accept a log transform
ar <- africa_recession %>% select(-y)
ac_mean <- sapply(ar, mean)
ac_sd <- sapply(ar, sd)
ac_max <- sapply(ar, max)
ac_min <- sapply(ar, min)

do_log_ind <- which(xor(ac_min < ac_mean - 
                          4*ac_sd, ac_max > ac_mean + 4*ac_sd) & ac_min > 0)

#Apply transform
ar <- ar%>%mutate_at(do_log_ind, log10)
#Rename these variables and count them
ar <- ar%>%rename_at(do_log_ind, list(~ paste(., "_log",sep="")))
log_count <- length(do_log_ind)
names(ar)[do_log_ind] %>% view()

#Check for discriminance - overlapping boxplots?
disc_plot <- ar%>%select(1:12) %>%
  mutate(y=y) %>%
  gather(k="key", value="value", -y) %>%
  ggplot(aes(y, value))+
  facet_wrap(~ key, nrow = 3, ncol = 4, scales = "free") +
  geom_boxplot()
disc_plot

#Standardise data
ars <- scale(ar)

# Look at collinearity
cor <- cor(ars) #using standardised version of dataset
image(x=seq(1,49), y=seq(1,49), xlab="", yaxt="n", xaxt='n', ylab="", z=as.matrix(cor),
      col = rev(RColorBrewer::brewer.pal(9, "RdBu")))
axis(side=1, at=seq(1,49), labels=names(ars), las=3, 
     cex.axis=0.5)
axis(side=2, at=seq(1,49), labels=names(ars), las=2, 
     cex.axis=0.5)

#Separate out test set
# Need a reasonable number of minority class in test set - hence 20% split
test_split <- 0.2 
set.seed(1,sample.kind = "Rounding")
test_index <- createDataPartition(y,1,test_split,list=FALSE)
ars <- as.data.frame(ars)
test_set <- ars[test_index,]
#Keep y vector separate in test set for final validation
test_set_y <- y[test_index]
train_set <- ars[-test_index,]
#Incude y vector in train set for caret implementation
train_set$y <- y[-test_index]
#Check prevalence in splits. (createData Partition should balance them)
prev_test <- round(mean(test_set_y == "recession"),4)
prev_train <- round(mean(train_set$y == "recession"),4)
#
# What is the best resampling method to balance the prevalence?
#
# Test various sample methods with knn
#
set.seed(5, sample.kind = "Rounding")
sampling_list <- c("none", "up", "smote", "rose")
sample_test <- lapply(sampling_list, function(sample_method){
  sm <- sample_method
  if (sample_method == "none") sm <- NULL #"none" is not accepted as sampling value
  ctrl <- trainControl(sampling=sm,
    method="repeatedcv", 
    number=3, # for maximum instances of minority class in each fold
    repeats = 10, #large to compensate for small number of folds
    summaryFunction = prSummary, # produces precision-recall stats
    classProbs = TRUE)
  tr <- train(y ~ .,
    method = 'knn',
    data=train_set,
    tuneGrid = data.frame(k = seq(2,20,2)), #determined from previous trial and error
    trControl=ctrl,
    metric="AUC") # area under precision-recall curve
  res <- as.data.frame(tr$results)
  res <- res %>% select(k,AUC,Recall,Precision)
  res$sample <- replicate(nrow(res),sample_method)
  res # Output dataframe of results for all k's so we can optimize for recall
})
sample_test <- do.call(rbind,sample_test) # combine list of dataframes into one
# Plot results
sample_plot <- sample_test[complete.cases(sample_test),] %>% 
  gather(key="stat", value="val", -k, -sample) %>%
  ggplot(aes(x = k))+
  facet_wrap(~ sample, nrow = 2, ncol = 2, scales = "fixed") +
  geom_smooth(aes(y = val, color = stat), size=0.5) +
  geom_point(aes(y = val, shape = stat, color=stat))
sample_plot
#
#Record best resampling method and k vakue
best_knn_method <- sample_test[which.max(sample_test$Recall),"sample"]
best_knn_k <- sample_test[which.max(sample_test$Recall),"k"]
#
# Use best_knn_method (up) in future for knn
#
# Do collinearities affect the model?
# Construct table of highly correlated variables from correlation matrix
# shave() removes upper right triangle and stretch() puts into long form
cor2 <- train_set %>% select(-y) %>%
  correlate() %>% shave() %>% stretch(na.rm = TRUE) %>%
  arrange(desc(r))
cor2 %>% filter(r > 0.75) %>% view()

#From this table remove variables progressively to the following limits:
# 0.98: -cda_log, -rdana_log, -rconna_log, -energy, -emp_log
# 0.95: -pl_gdpo, -pl_da, -pl_c_log, -ctfp, -cn_log, -ck_log,
#      -metals_minerals, -agriculture
# 0.90: -rnna_log, -energy_change
# 0.75: -excl_energy, -pl_m_log, -pl_x_log,-excl_energy_change, 
#      -metals_minerals_change,-agriculture_change, -rwtfpna_log, -fish,
#      -pop_log
#
# 4 derivative training sets created, with reduced variables:

train_set98 <- train_set%>%select(-cda_log, -rdana_log, -rconna_log, 
                                  -energy, -emp_log)
train_set95 <- train_set98%>%select(-pl_gdpo, -pl_da, -pl_c_log, -ctfp, -cn_log, 
                                    -ck_log, -metals_minerals, -agriculture)
train_set90 <- train_set95%>%select(-rnna_log, -energy_change)
train_set75 <- train_set90%>%select(-excl_energy, -pl_m_log, -pl_x_log,
                                    -excl_energy_change, -metals_minerals_change,
                                    -agriculture_change, -rwtfpna_log, -fish,
                                    -pop_log)

#Test performace of these training sets on knn with k=12, "up" sampling

set_list <- list() # list of different training sets with reduced variables
set.seed(7, sample.kind = "Rounding")
set_list[[1]] <- list(cut_off=1.00,set=train_set)
set_list[[2]] <- list(cut_off=0.98,set=train_set98)
set_list[[3]] <- list(cut_off=0.95,set=train_set95)
set_list[[4]] <- list(cut_off=0.90,set=train_set90)
set_list[[5]] <- list(cut_off=0.75,set=train_set75)

ctrl <- trainControl(sampling=best_knn_method, # parameters as previously determined
                     method="repeatedcv", 
                     number=3,
                     repeats = 10,
                     summaryFunction = prSummary, 
                     classProbs = TRUE)

variable_cut <- lapply(set_list, function (sets) {
  tr <- train(y ~ ., 
        method = "knn",
        data = sets$set, 
        trControl = ctrl,
        tuneGrid = data.frame(k = best_knn_k),
        metric = "AUC")
  res <- as.data.frame(tr$results)
  res <- res %>% select(AUC, Recall, Precision)
  res$Cut_off <- sets$cut_off
  res # dataframe output of training results
})
variable_cut <- do.call(rbind, variable_cut) # Combine list of dataframe outputs into one
# Plot results  
var_plot <- variable_cut %>%
  gather(key="stat",value="val", -Cut_off) %>%
  ggplot(aes(x = Cut_off, y = val)) + 
  geom_point(aes(color=stat, shape=stat)) +
  geom_smooth(aes(y = val, color = stat), size=0.5) +
  scale_x_reverse()
var_plot
#
#
#Look at PCA - does that help our model?
#
pca<-prcomp(select(train_set, -y)) #Use the 95% reduced-variable training set
# calculate cumulative sum of variance vector and add in zero point
c_var <- c(0, cumsum(pca$sdev^2 / sum(pca$sdev^2) ))
# plot the result
cum_pca_plot <- ggplot(mapping = aes(x=seq(0, ncol(pca$x)), 
                           y=c_var)) +
  geom_point() +
  geom_line() +
  xlab("PC") +
  ylab("Cumulative variance")
cum_pca_plot

# How do input variables map to PCs? Look at the rotaion matrix from prcomp:
image(x=seq(1,49), y=seq(1,49),t(pca$rotation), col = rev(RColorBrewer::brewer.pal(9, "RdBu")), 
      xlab="PC", yaxt="n", xaxt='n',ylab="")
axis(side=2, at=seq(1,49), labels=rownames(pca$rotation), las=1, 
     cex.axis=0.5)
axis(side=1, at=seq(1,49), labels=seq(1,49), las=1, cex.axis=0.5)
#
#Test out levels of PCA cut-off - does it affect our model?
#
# Try 100%, 98%, 95%, 90%, 80%
#
set.seed(5,sample.kind = "Rounding")
pca_test <- lapply(c(1.0,0.98,0.95, 0.90, 0.8), function(threshhold){
  ctrl <- trainControl(sampling=best_knn_method, 
    method="repeatedcv", # parameters as previously determined
    number=3,
    repeats = 10,
    summaryFunction = prSummary,
    classProbs = TRUE,
    preProcOptions = list(thresh=threshhold)) # set PCA cutoff to be applied
  tr <- train(y ~ .,
    method = 'knn',
    data=train_set,
    tuneGrid = data.frame(k = best_knn_k),
    trControl=ctrl,
    preProcess=c("pca"), # apply PCA cutoff
    metric="AUC")
  res <- as.data.frame(tr$results)
  res <- res%>%select(AUC, Recall, Precision)
  res$Threshhold <- threshhold
  res #Output dataframe of results
} )
pca_test <- do.call(rbind, pca_test) # Combine list of dataframes into one
#Plot results
pca_plot <- pca_test %>% 
  gather(key="stat",value="val", -Threshhold) %>%
  ggplot(aes(x = Threshhold, y = val)) + 
  geom_point(aes(color=stat, shape=stat)) + 
  geom_smooth(aes(y = val, color = stat), size=0.5) +
  scale_x_reverse()
pca_plot

### TRY RANDOMFORESTS 
# with different resampling techniques and a range 
# of nodesizes: 10 to 60.
#
# THE FOLLOWING BLOCK OF CODE WULL RUN FOR A LONG TIME!
#
nodes <- seq(10,60,10) # Minimum permitted size of a node
set.seed(7,sample.kind = "Rounding")
# Same list of sampling methods is used: "sampling_list"
rf_sample_test <- lapply(sampling_list, function(sample_method){
  # loop for different sample methods
  sm <- sample_method
  if (sample_method == "none") sm <- NULL # "none" is not accepted as a value of sampling parameter
  ctrl <- trainControl(sampling=sm, 
    method="repeatedcv",
    number=3,  #parameters as previously determined 
    repeats = 10,
    summaryFunction = prSummary, 
    classProbs = TRUE)
  rf_node_test <- lapply(nodes, function(nsize){
    #inner loop for different nodesizes
    tr <- train(y ~ .,
      method = 'rf',
      data=train_set,
      tuneGrid = data.frame(mtry = seq(3,10)), #determined by trial and error
      trControl=ctrl,
      nodesize = nsize,
      metric="AUC")
    res <- as.data.frame(tr$results)
    res <- res[which.max(res$Recall),] %>% select(mtry, AUC, Precision, Recall)
    res$sample <- sample_method
    res$nodesize <- nsize
    res  #Output dataframe of results  
  })
  rf_node_test <- do.call(rbind,rf_node_test) #Combine dataframes for each sampling method
  rf_node_test
}) #PLEASE BE PATIENT WHILE THIS CODE BLOCK RUNS!!!
rf_sample_test <- do.call(rbind,rf_sample_test) #Combine dataframes for all sampling methods
#plot results
sample_plot_rf <- rf_sample_test %>%
  gather(key = stat, value = val, -mtry, -sample, -nodesize) %>%
  ggplot(aes(x = nodesize))+
  facet_wrap(~ sample, nrow = 2, ncol = 2, scales = "fixed") +
  geom_smooth(aes(y = val, color = stat), size=0.5) +
  geom_point(aes(y = val, color = stat, shape=stat))
sample_plot_rf
#
# Record best method, nodesize and mtry
best_rf <- which.max(rf_sample_test$Recall)
best_rf_method <- rf_sample_test[best_rf,"sample"]
best_rf_nodesize <- rf_sample_test[best_rf,"nodesize"]
best_rf_mtry <- rf_sample_test[best_rf,"mtry"]
#
#Does dropping variables help us with rf?
#
set.seed(7, sample.kind = "Rounding")
ctrl <- trainControl(sampling=best_rf_method, 
                     method="repeatedcv", 
                     number=3, #Parameters previosuly determined
                     repeats = 10,
                     summaryFunction = prSummary, 
                     classProbs = TRUE)
#use previously defined set_list
rf_variable_cut <- lapply(set_list, function (sets) {
  tr <- train(y ~ ., 
              method = "rf",
              data = sets$set, 
              trControl = ctrl,
              nodesize = best_rf_nodesize, 
              tuneGrid = data.frame(mtry = best_rf_mtry), 
              metric = "AUC")
  res <- as.data.frame(tr$results)
  res <- res %>% select(AUC, Recall, Precision)
  res$Cut_off <- sets$cut_off
  res
})
rf_variable_cut <- do.call(rbind, rf_variable_cut)
# Plot results  
rf_var_plot <- rf_variable_cut %>%
  gather(key="stat",value="val", -Cut_off) %>%
  ggplot(aes(x = Cut_off, y = val)) + 
  geom_point(aes(color=stat, shape =stat)) + 
  geom_smooth(aes(y = val, color = stat), size=0.5) +
  scale_x_reverse()
rf_var_plot
#Will PCA help random forests?
set.seed(5, sample.kind = "Rounding")
pca_test_rf <- lapply(seq(0.5, 1.0, 0.1), function(threshhold){
  ctrl <- trainControl(sampling=best_rf_method,
                       method="repeatedcv",
                       number=3, #Parameters previosuly determined
                       repeats = 10,
                       summaryFunction = prSummary,
                       classProbs = TRUE,
                       preProcOptions = list(thresh=threshhold)) #apply PCA cut-offs
  tr <- train(y ~ .,
              method = 'rf',
              data=train_set,
              tuneGrid = data.frame(mtry = best_rf_mtry),
              trControl=ctrl,
              nodesize = best_rf_nodesize,
              preProcess=c("pca"), #apply PCA rotation
              metric="AUC")
  res <- as.data.frame(tr$results)
  res <- res %>% select(AUC, Recall, Precision)
  res$Threshhold <- threshhold
  res
} )
pca_test_rf <- do.call(rbind, pca_test_rf)
# plot results  
pca_plot_rf <- pca_test_rf %>% 
  gather(key="stat",value="val", -Threshhold) %>%
  ggplot(aes(x = Threshhold, y = val)) + 
  geom_point(aes(color=stat, shape=stat)) + 
  geom_smooth(aes(y = val, color = stat), size=0.5) +
  scale_x_reverse()
pca_plot_rf

# Try on test data
#
# Build final knn model
#
set.seed(1,sample.kind = "Rounding")
ctrl_knn <- trainControl(sampling=best_knn_method,
                        method="repeatedcv", 
                        number=3,
                        repeats = 10,
                        summaryFunction = prSummary, 
                        classProbs = TRUE)

knn_final <- train(y ~ ., 
                   data=train_set,
                   method='knn',
                   trControl=ctrl_knn,
                   tuneGrid = data.frame(k = best_knn_k),
                   metric = "AUC")
#test_set95 <- test_set%>% select(-cda_log, -rdana_log, -rconna_log, 
#                                 -energy, -emp_log,-pl_gdpo, -pl_da, -pl_c_log, -ctfp,
#                                 -cn_log, -ck_log, -metals_minerals, -agriculture)
y_hat_knn <- predict(knn_final,test_set)
cf_knn <- confusionMatrix(y_hat_knn, test_set_y)
# Convert factors to (0,1) for AUC calculation
y_auc_pred_knn <- ifelse(y_hat_knn == "recession",0,1)
y_auc <- ifelse(test_set_y == "recession",0,1)
auc_knn <- PRAUC(y_auc_pred_knn,y_auc)
#
# and final rf model
#
ctrl_rf <- trainControl(sampling=best_rf_method,
                           method="repeatedcv", 
                           number=3,
                           repeats = 10,
                           summaryFunction = prSummary, 
                           classProbs = TRUE)
rf_final <- train(y ~ .,
            method = 'rf',
            data=train_set,
            tuneGrid = data.frame(mtry = best_rf_mtry),
            trControl=ctrl_rf,
            nodesize = best_rf_nodesize,
            metric="AUC")
y_hat_rf <- predict(rf_final,test_set)
cf_rf <- confusionMatrix(y_hat_rf, test_set_y)
# Convert factors to (0,1) for AUC calculation
y_auc_pred_rf <- ifelse(y_hat_rf == "recession",0,1)
auc_rf <- PRAUC(y_auc_pred_rf,y_auc)
#
# try an ensemble with rule y_hat_rf or y_hat_cnn
#
y_hat_ensemble <- ifelse(y_hat_knn == "recession" | y_hat_rf == "recession",
                     "recession", 
                     "growth")
y_hat_ensemble <- factor(y_hat_ensemble, levels=c("recession","growth"), ordered=TRUE)

cf_ensemble <- confusionMatrix(y_hat_ensemble, test_set_y)
y_auc_pred_ensemble <- ifelse(y_hat_ensemble == "recession",0,1)
auc_ensemble <- PRAUC(y_auc_pred_ensemble,y_auc)
#
#recall no better, precision slightly worse
#Look at importance:

imp <- as.data.frame(importance(rf_final$finalModel))
imp <- data.frame(variable = row.names(imp), imp)
imp <- imp %>% arrange(desc(MeanDecreaseGini))