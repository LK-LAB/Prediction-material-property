
#### Written by Jeongrae Kim in KIST

#### Load libray
library(PeriodicTable)
library(stringr)
library(readr)

#### Setting working directory
setwd("D:/6_SS_2_ppp/EEAE")

#### Load cif file
dir_cif <- "./Data_cif_sample/"
vList_cif <- list.files(dir_cif)
vList_name <- str_sub(vList_cif, 1, -5)

#### Load element
data("periodicTable")
PT <- periodicTable[-1,]
vElement <- PT$symb

#### Make matrix 
df_MP <- as.data.frame(matrix(0,length(vList_cif),(length(vElement)+1)))
colnames(df_MP) <- c("name",vElement)
df_MP$name <- vList_name
for(cif in 1:nrow(df_MP)){
    M_name <- df_MP$name[cif]
    t <- readLines(paste(dir_cif, M_name, ".cif", sep=''))
    vElement_ <- strsplit(gsub("'", "", gsub("_chemical_formula_sum", "", t[12])), " ")[[1]]
    vElement <- vElement_[vElement_ != ""]
    for(Ele in 1:length(vElement)){
        en <- vElement[Ele]
        n <- parse_number(en)
        e <- gsub(n, "", en)
        df_MP[cif,colnames(df_MP)==e] <- n
    }
}

#### Normalization: TF
df_TF <- df_MP[,-1]
cRowsum <- as.vector(rowSums(df_TF))
for(r in 1:nrow(df_TF)){
    df_TF[r,] <- df_TF[r,]/cRowsum[r]
}
df_TF_vColsum <- as.vector(colSums(df_TF != 0))
df_CM_TF <- df_TF[,df_TF_vColsum != 0]

#### Normalization: iDF
df_CM_IDF <- as.data.frame(matrix(0, dim(df_CM_TF)[1], dim(df_CM_TF)[2]))
colnames(df_CM_IDF) <- colnames(df_CM_TF)
vColsum <- as.vector(colSums(df_CM_TF != 0))
vTotalRow <- nrow(df_CM_TF)
for(c in 1:ncol(df_CM_TF)){
    df_CM_IDF[,c] <- log(vTotalRow/(vColsum[c]))
}
#### Normalization: TF*iDF
df_CM_TFIDF <- df_CM_TF*df_CM_IDF
head(df_CM_TFIDF)
dim(df_CM_TFIDF)

#### Save: csv type
write.csv(df_CM_TFIDF, file="1_CompositionMatrix_TFIDF.csv", row.names=FALSE)

#### Save: element name
vElement <- colnames(df_CM_TFIDF)
write.table(vElement, file="1_CompositionMatrix_element.txt", col.names = FALSE, row.names=FALSE, sep=",")
