#!/usr/bin/env Rscript
library(boot)

args = commandArgs(trailingOnly=TRUE)
if(length(args) != 1) {
      stop('Provide the results CSV file.')
}

df <- read.csv(file=args[1], header=TRUE, sep=',')

f <- function(data, indices){
    s <- colSums(data[indices, ]) * 100 / nrow(data)
    return (s)
}

r_boot <- boot(data=df, statistic=f, R=10000)
i <- 1 
for (n in names(df)){
    CI <- boot.ci(r_boot, type='perc', index=i)
    cat(sprintf('%s %.2f [%.2f -- %.2f]\n', n ,CI['t0'], CI['percent'][[1]][4], CI['percent'][[1]][5]))
    i <- i + 1
}

