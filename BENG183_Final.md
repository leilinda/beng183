# Machine Learning Applications in Genomics
### BENG 183 Fall 2018 
<b>Xin Yi Lei</b>  

1. [Introduction](#1)   
2. [Introduction to Supervised vs. Unsupervised Learning](#2)  
3. [Model Selection and Optimization](#3)   
4. [Supervised Learning](#4)  
5. [Unsupervised Learning](#5)  
6. [Sources](#6)  

## I. Introduction<a name="1"></a>

Machine learning is an large branch of computer science and mathematics that involves algorithms and method development to aid in predictive modelling. Within the last few decades, large amounts of data have become available, and machine learning algorithms and techniques have enabled Bioinformaticians to utilize the datasets to predict, model, and make sense of the inflow of data. Machine learning has been used in the interpretation of large genomic data sets and the annotation of genomic sequence elements based on DNA pattern recognition. Since then, machine learning approaches have extended to other types of data, such as RNA-Seq differential gene expression, ChIP-Seq and chromatin annotation. 

## II. Supervised Learning vs. Unsupervised Learning <a name="2"></a>

Machine learning can be separated into two types - supervised and unsupervised learning. 

#### a. Supervised Learning
Supervised learning involves training a model with already labelled data in order to make predictions about unlabelled examples [1]. After the model learns the general properties of the data, it uses those learned properties to infer new information in the unlabelled data. 

There are three elements to supervised learning:

- Training input: a collection of data points, each with an associated label
- Post-training input: a collection of data points
- Post-training output: predicted label for the unlabelled data points

#### b. Unsupervised Learning
Unsupervised learning is trying to find structure in a data set without any labels [1]. Note that for this type of learning, there is no training input, so structure often involves clustering and dimensionality reduction to ease interpretation of large and complex genomic datasets.

- Input: a collection of data points without labels 
- Output: predicted label for the unlabelled data points

## III. Model Selection and Optimization  <a name="3"></a>  
![](/final_figures/tvt.png)  

Model selection and optimization involves splitting the data into training, validation, and testing in order to ensure the model is both specific and predictive, but generalizable.  

The model is initially fit on a <b>training</b> dataset, that is a set of examples used to fit the parameters. Then, the fitted model is tested on the <b>validation</b> dataset, which offers an unbiased evaluation of the model and finetunes the hyperparameters and to make sure there is no overfitting. Finally, the <b>test</b> dataset is used to provide an unbiased evaluation of a final model fit on the training dataset.  

## IV. Supervised Learning Applications<a name="4"></a> 
Examples of supervised learning in genetics and genomics range from DNA pattern recognition to text-mining. In this section, we will discuss two prominent examples of supervised learning, and demonstrate some applications and their influence to the field.

#### a. Hidden Markov Models  
One of the most pervasive examples of supervised learning is the Hidden Markov Model, hereinafter referred to as the HMM. HMMs are a type of Bayesian network that use transition and emission probabilities to interpret the state probability of a dataset. HMMs have been used in bioinformatic modeling from chromatin state discovery [2] to gene finding [1]. Here, we present a simple gene-finding HMM used to capture the properties of a protein-coding gene.  
![](/final_figures/gene_finding.jpg)  

This model requires as input a training set of labeled DNA sequences. Labels for such a model may include the start and end coordinates of a gene, splice sites, and UTR. The model uses this training data to learn general properties of genes, including the DNA pattern near donor and acceptor splice sites, the expected length distributions for each element, and patterns within elements, such as typical codons in an exon vs. an intron. The trained model can then use these learned properties to identify novel genes that resemble the genes in the training set.  

Note certain limitations to this type of model, namely that it is incapable of identifying overlapping genes or multiple isoforms of the same gene. Similar limitations exist for other applications of HMMs, such as in topologically associating domain (TAD) discovery and directionality index creation in Hi-C data, wherein a simple HMM as above cannot find a hierarchy of TADs. For biological questions like this, more complex models are useful.  

#### b. Linear Regression  
Another commonplace example of supervised learning in genomics is linear regression, which can be used in population genomics and medical genetics to perform single SNP association testing through optimizing the sum of the squared differences in a dataset. Notice how this dataset can be modeled in three ways - additive, dominant, and recessive. Each model attempts to estimate the proportion of phenotypic variation, in this case, cholesterol, explained by the SNP, and uses the coefficient of determination, or R^2 value, to assign confidence to that estimation [3]. 
![](/final_figures/regression.JPG)  
One approach to choosing a statistical model is to choose the simplest model that has the right parameters and right assumptions based on our biological data, and work toward building rigor and complexity. Thus, it is important to understand what the assumptions of each model are, and how it relates to the problem we are trying to solve. In the additive model, for example, by putting a line to the dataset, we are exploring the phenomenon that the number of minor alleles, or major alleles, is correlated with phenotypic variation, and as such the two homozogytes are equally different to the heterozygote. Such distinctions are important when performing association studies, e.g. for purposes related to drug target discovery.  

## V. Unsupervised Learning Applications<a name="5"></a>  
Finally, bioinformaticians often work with unlabelled data and use various clustering and dimensionality reduction algorithms to make sense of it. Here, we will discuss two such algorithms and their uses in the field.  

#### a. Hierarchical Clustering  
![](/final_figures/dendrogram.JPG)  
Hierarchical clustering is useful in clustering datapoints to determine a similarity relationship within clusters and between clusters. The algorithm for hieararchical clustering is as follows:  
```
1. First, what determines what makes some points more “similar” to each other than other points. For example, we could use distance between points as a basis for similarity  
2. The similarity between all pairs of points is calculated
3. Each profile is placed in its own cluster
4. The two most similar cluster are paired together into a new cluster (While doing this we create a dendrogram, where these two points are connected by a branch)
5. Similarity is calculated using a specified clustering method (Examples are UPGMA, Complete Linkage and Single Linkage)
6. Repeat steps the above two steps until everything is in one cluster
```
Hiearchical clustering has applications including quality checking (do technical/biological replicates cluster together?) or in evolutionary genomics, such as phylogenetic tree inference.  

#### b. K-means Clustering  
![](final_figures/kmeans.JPG)  
K-means clustering offers a simple alternative method of aggregating datapoints for further analysis. The algorithm for k-means clustering is as follows:  
```
1. Begin with predetermined choice of clusters K, which are represented by centroids. Iterate the following two steps.
2. For each data point, find the closest mean vector and assign the object to the corresponding cluster
3. For each cluster, update its mean vector based on all the points that have been assigned to it
4. Terminate after predetermined number of iterations OR after a certain percentage of data points converge inside a cluster. (Alternatively, when nothing changes, meaning no points are assigned to different clusters compared to the previous iteration, we stop.)
```  
K-means clustering can be used to cluster gene expression profiles to predict gene function or cluster patient samples to predict clinical characteristics [4]. Altogether, both clustering methods offer valuable information in inferring and modeling the behavior of biological occurrences. One key difference between the two methods is that hierarchical clustering is determinate, that is it will always result in the same solution, whereas k-means depends on random initialization, which may change the solution. Another difference is that hierarchical clustering allows the scientist to understand the hierarchy of the dataset, whereas k-means rigidly assigns clusters without offering an understanding of their relationship within and to each other.  

Altogether, machine learning is picking up speed in a wide array of bioinformatic applications, and it is critical to assess the considerations and recurrent challenges in such applications as they approach clinical and widespread usage. Nonetheless, such techniques offer a powerful basis from which to provide insight toward discovery in genomics.  

# Sources outside of class<a name="6"></a>  
[1] Libbrecht M.W., Noble W. S. Machine Learning Applications in Genetics and Genomics. Nature Reviews Genetics. 16, (2015) 321-332.  
[2] Ernst, Jason and Manolis Kellis. “ChromHMM: automating chromatin-state discovery and characterization” Nature methods vol. 9,3 215-6. 28 Feb. 2012, doi:10.1038/nmeth.1906  
[3] Thornton, Timothy and Michael Wu. "Genetic Association Testing with Quantitative Traits". Summer Institute in Statistical Genetics 2015 Lecture 2.  
[4] Hong, Pengyu. "Clustering Algorithms, Bioinformatics Data Analysis and Tools". http://www.ibi.vu.nl/teaching/masters/bi_tools/2008/tools_lec4_2008_handout.pdf
