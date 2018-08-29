### get wordVectors package from github if needed 
if(!require("wordVectors")) {
  
  install.packages("devtools")
  devtools::install_github("bmschmidt/wordVectors")
  
}

library(Matrix) ### for sparse matrices
library(wordVectors) ### cosine similarity
library(data.table) ### quickly load embeddings with fread()
library(parallel) ### faster graph construction

### functions ###

### finds nearest neighbours
get_nn <- function(i, nn, embeddings) {
  
  temp  <- acos(-pmax(pmin(cosineSimilarity(t(embeddings[i,]), embeddings), 1), 0))/pi
  index <- sort.list(temp,decreasing = T)[2:(nn+1)]
  
  cbind(i, index, temp[index])
  
}

### running the random walk ###
# seeds = vector of seed words
# transition_matrix = the output of build_graph()
# beta = beta parameter, "damping factor"
# tol = tolerance for update of ranking vector, usually only requires less than 10 iterations 

random_walk <- function(seeds, transition_matrix, beta = 0.5, tol = 1e-6) {
  
  voc <- colnames(transition_matrix)
  
  if(!all(seeds %in% colnames(transition_matrix))) {stop(paste("Seeds", seeds[!seeds %in% colnames(transition_matrix)], "are not in graph"))}
  
  P <- rep(1, length(voc))/length(voc)
  S <- ifelse(voc %in% seeds, 1/length(seeds[seeds %in% voc]), 0)*(1 - beta)
  
  repeat {
    
    P.new <- as.vector(beta*transition_matrix%*%P + S)
    
    if(sum(abs(P.new - P)) < tol) {P <- P.new;break}
    
    P <- P.new
    
  }
  
  return(P)
  
}

### graph construction, can run in parallel for speed (but this requres duplicating the embeddings in RAM for each core used)
### em = embeddings, rows correspond to words, columns to embeddings vectors
### vocab = a vector of words of length equal to the number words with embeddings, usually rownames(embeddings)
### nn = number of nearest neighbours to include in graph
### cores = if greater than 1, builds the graph in parallel for speed
build_graph <- function(em, vocab, nn = 25, cores = 6) {
  
  if(!nrow(em)==length(vocab)) {stop("number of embeddings and number of words in vocabulary do not match")}
  
  if(cores > 1) {
    
    cl <- makeCluster(getOption("cl.cores", cores))
    
    clusterExport(cl, c('em', 'get_nn', 'cosineSimilarity'))
    
    G <- parLapply(cl, 1:length(vocab), fun = get_nn, nn = 25, embeddings=em)
    
    stopCluster(cl)
    
  } else {
    
    G <- lapply(1:length(vocab), FUN = get_nn, nn = 25, embeddings=em)
    
  }
  
  E <- do.call(rbind, G)
  E <- sparseMatrix(i = E[,1], j = E[,2], x = E[,3], dims = c(length(vocab), length(vocab)), dimnames = list(vocab, vocab))
  
  D <- sparseMatrix(i=1:ncol(E),j=1:ncol(E),x=1/sqrt(rowSums(E)))
  
  Tr <- D%*%E%*%D
  
  dimnames(Tr) <- dimnames(E)
  
  return(Tr)
  
}

### my quick function for bootstrapping CIs
### pos_seeds = positive seed words
### neg_seeds = negative seed words
### trans_matrix = a transition matrix from build_graph()
### boot_seeds = number of random seed words selected in each bootstrap iteration
### boot_n = number of bootstrap iterations
### ... = other arguments to random_walk() beta and tol

bootstrap_fnc <- function(pos_seeds, neg_seeds, trans_matrix, boot_seeds, boot_n, ...) {
  
  pos <- list()
  neg <- list()
  
  for(i in 1:boot_n) {
    
    pos[[i]] <- random_walk(seeds = sample(pos_seeds, boot_seeds), transition_matrix = trans_matrix, ...)
    neg[[i]] <- random_walk(seeds = sample(neg_seeds, boot_seeds), transition_matrix = trans_matrix, ...)
    
  }
  
  final <- list()
  
  for(i in 1:boot_n) {
    
    final[[i]] <- pos[[i]]/(pos[[i]]+neg[[i]])
    
  }
  
  dat <- do.call(cbind, final)
  
  output <- data.frame(Word=colnames(Tr), Score=apply(dat, 1, mean), Score.SD=apply(dat, 1, sd),stringsAsFactors = F)
  
  return(output)
  
}

### a brief example using 3000 embeddings

### some open-source fasttext embeddings
# download.file("https://s3-us-west-1.amazonaws.com/fasttext-vectors/wiki-news-300d-1M-subword.vec.zip")
#### these embeddings have an absolute ton of crap, but still serve to show the implementation at work

em <- fread('wiki-news-300d-1M-subword.vec', header = F, data.table = T, skip = 1, nrows = 3000, quote = "", sep = " ")

### need matrix for cosine distance calculation
vocab <- em$V1

em$V1 <- NULL

em <- as.matrix(em)


### make the graph and transition matrix on 2 cores
Tr <- build_graph(em = em, vocab = vocab, cores = 2, nn = 25)


#### get some seed words
conc <- fread('http://www.humanities.mcmaster.ca/~vickup/Concreteness_ratings_Brysbaert_et_al_BRM.csv')
conc <- conc[Word %in% vocab]

### take 20 most concrete and abstract as seeds
concrete <- head(conc[order(conc$Conc.M, decreasing = T),], 20)
abstract <- tail(conc[order(conc$Conc.M, decreasing = T),], 20)

test <- bootstrap_fnc(pos_seeds = concrete$Word, neg_seeds = abstract$Word, trans_matrix = Tr, boot_seeds = 15, boot_n = 50, beta = 0.99, tol = 1e-6)

test <- test[order(test$Score,decreasing=T),]

head(test, 100)
tail(test, 100)

compare <- merge(test, conc, by.x = "Word", by.y = "Word")

with(compare[!compare$Word %in% c(concrete$Word, abstract$Word),], {print(cor.test(Conc.M, Score));plot(Conc.M, Score)})

