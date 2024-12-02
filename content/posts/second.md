+++
date = '2024-10-30T00:54:41+05:30'
draft = false
title = "Basics of NLP"
+++








You must have used ChatGPT, but how does it work? How does Google Translate work? They all work under the study of **`Natural Language Processing.`**

<aside>

Natural Language Processing (NLP) is a field of artificial intelligence focused on enabling machines to understand, interpret, and generate human language. It combines computational linguistics, machine learning, and deep learning to bridge the gap between human communication and computer understanding.

</aside>

---
This blog has four parts 
- **`Text Preprocessing`** - Lowercasing, Tokenisation, Stop-word Removal, Stemming, Lemmatization
- **`Regex`** - Basics and some examples
- **`Frequencies`** - BoW, TF, IDF, TF-IDF
- **`Word Embeddings`** - Word2Vec, GloVe

---

# `Text Preprocessing`

We will be using the NLTK library for this task, make sure import it earlier on using `import nltk`

## `Lowercasing`

This step is obvious, we do this to ensure consistency. It also reduces the complexity of the text data we are going to input. We can simply just convert every character in the test data to its lower case form.

```python
text=#your input data
text=text.lower()

```

## `Tokenization`

Imagine you want to teach your kid to study English, instead of teaching him Shakespeare, you will first teach him alphabets, then words then sentences.

Here your child is the machine, and we break down sentences for it to learn so that the sentence is broken into meaningful tokens, which still carry the original essence of context. This makes pattern recognition easier.

Let’s say we have a sentence `I am stuipd`. We can break it down to `["I","am","stupid"]` . This is word tokenization, which breaks long sentences into individual words.

Now let’s go down one step further. ['I', ' ', 'a', 'm', ' ','s', 't', 'u', 'p', 'i', 'd'] . This kind of tokenization is called character tokenization, mostly used for spelling correction tasks.
We can also break words neuralnets to ["neural","nets"].This kind of tokenization is termed as subword tokenization. This is useful for languages that form meaning by combining smaller tokens.

```python

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')
tokens = word_tokenize("I am neuralnets.")
print("Tokens:", tokens)
```

## **`Stop-word Removal`**

What is a stop-word? They are usually the words that don’t contribute any meaning or context to the sentence. For example, the word *the* doesn’t bring out any significance to the sentence, hence we can just remove it from the input data. But words which have important context to the sentence, like for example, *history*  can’t be termed as stop-words.

```python
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english') 
#this forms a set of stop words
words=[word for word in tokens if word not in stop_words]
#this stores all words in tokens, that are not stopwords
```

## **`Stemming`**

Let us consider this sentence, The leaves are falling, have fallen, but will leave behind beauty

With stemming, what we do is chop down the sentence into their root forms, which might not make any sense and lose the meaning of the word.

![https://trite-song-d6a.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fcc95b4dc-10fa-4b10-a476-92cd53dd254d%2F6114946a-89bf-4cf7-bd26-748418ae0310%2Fimage.png?table=block&id=1440af77-bef3-805f-8824-c87adf559760&spaceId=cc95b4dc-10fa-4b10-a476-92cd53dd254d&width=640&userId=&cache=v2](https://trite-song-d6a.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fcc95b4dc-10fa-4b10-a476-92cd53dd254d%2F6114946a-89bf-4cf7-bd26-748418ae0310%2Fimage.png?table=block&id=1440af77-bef3-805f-8824-c87adf559760&spaceId=cc95b4dc-10fa-4b10-a476-92cd53dd254d&width=640&userId=&cache=v2)

```python
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokeniz
text = "The leaves are falling, have fallen, but will leave behind beauty."
words = word_tokenize(text)
stemmer = PorterStemmer()
stems = [stemmer.stem(word) for word in words]
print("Stemming Results:", stems)
```

## **`Lemmatization`**

Consider the earlier sentence.
What we do here, is we chop down the sentence into their base dictionary word, which considers the meaning and context of the word, hence not making a word, that makes no sense.

![https://trite-song-d6a.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fcc95b4dc-10fa-4b10-a476-92cd53dd254d%2F1885e625-4d43-48db-a1eb-7f4cbd1abcdc%2Fimage.png?table=block&id=1440af77-bef3-80b4-8dcc-cbccadc8726f&spaceId=cc95b4dc-10fa-4b10-a476-92cd53dd254d&width=600&userId=&cache=v2](https://trite-song-d6a.notion.site/image/https%3A%2F%2Fprod-files-secure.s3.us-west-2.amazonaws.com%2Fcc95b4dc-10fa-4b10-a476-92cd53dd254d%2F1885e625-4d43-48db-a1eb-7f4cbd1abcdc%2Fimage.png?table=block&id=1440af77-bef3-80b4-8dcc-cbccadc8726f&spaceId=cc95b4dc-10fa-4b10-a476-92cd53dd254d&width=600&userId=&cache=v2)

```python
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
text = "The leaves are falling, have fallen, but will leave behind beauty."
words = word_tokenize(text)
stemmer = WordNetLemmatizer()
lemmas = [lemmatizer.lemmatize(word, pos="v") for word in words])
print("Lemmatization Results:", lemmas)
```

---

# `Regular Expressions(Regex)`

A regular expression (shortened as regex) is a sequence of characters that specifies a search pattern in text. Basically it’s a way of finding and searching stuff in a string.

### `How to write Regex?`

→ First we import the regex module with `import re`

→ Then we need to create a regex object with `re.compile`

→ We need to pass our input string into the Regex object using `search()` , which returns an object

→ Finally call the `group()` function to return a string output from the object.

### `Common Regex Symbols`

![image.png](Basics%20of%20NLP%2014c17bf3de1e80a5bbeaf3a79696a82d/image.png)

### `Example`

```python
phone_num_regex = re.compile(r'\d\d\d\d-\d\d\d\d\d\d')

mo = phone_num_regex.search('My number is 9834-872919.')

print(f'Phone number found: {mo.group()}')
#this will output the number in the string
```

Since this topic is very practical, and should be learnt through practice, you’re suggested to try examples out on your own to get the grasp of regex! Check out -

[https://docs.python.org/3/howto/regex.html](https://docs.python.org/3/howto/regex.html)

---

# `Frequencies`

### `Bag of Words(BoW)`

It is nothing but a simple way to represent text data. We don’t care about the order of words not the grammar, we just care about what words appear in the sentence. It’s like putting words in a bag and then randomly counting each type of word.

Firstly, we need to create a vocabulary, that is all the unique words in our dataset. Then we count how often each word occurs in the vocabulary.

Let’s say we have sentences 

S1 : `I like pizza`

S2 : `I do not like pizza`

Our vocabulary will be ["I", "like", "pizza", "do", "not"]

Now we count the number of appearances

| **Word** | **Sentence 1** | **Sentence 2** |
| --- | --- | --- |
| **I** | 1 | 1 |
| **Like** | 1 | 1 |
| **pizza** | 1 | 1 |
| **do** | 0 | 1 |
| **not** | 0 | 1 |
| **Total Words** | 3 | 5 |

The BoW representation is :

1. Sentence 1 : `[1,1,1,0,0]`
2. Sentence 2 : `[1,1,1,1,1]`

The BoW model represents each sentence as a vector, where each dimension is a word in the vocabulary and the value in it represents the count of that word.

### `Term Frequency(TF)`

It measures how often a specific word appears in a document, compared to the total number of words in that document. Instead of counting like in BoW, TF normalises the counts, thus for larger datasets, we can see how important a word is in context of the document. It sets the stage for more advanced models to come.

*The formula for TF is:*

$TF(w)=\frac{count \space of \space w \space in \space document}{Total\space number \space of \space words\space in \space document}$

We will be using the same example like the last time.

| **Word** | **TF of Sentence 1** | **TF of Sentence 2** |
| --- | --- | --- |
| **I** | 1/3 = 0.333 | 1/5 = 0.2 |
| **Like** | 1/3 = 0.333 | 1/5 = 0.2 |
| **pizza** | 1/3 = 0.333 | 1/5 = 0.2 |
| **do** | 0 | 1/5 = 0.2 |
| **not** | 0 | 1/5 = 0.2 |
| **Total Words** | 3 | 5 |

### `Inverse Document Frequency(IDF)`

While TF told us about the importance of a word in a document, IDF tells us about how rare or unique a word is in the document. Using this, we can often eliminate words that carry very less context, rather than the rare words which might carry huge context. Words that appear in every document get a lower IDF (close to 0). Words that appear in fewer documents get a higher IDF.

*The formula for IDF  is :*

$IDF(w)=log(\frac{Total\space number \space of \space documents}{Number\space of \space documents\space having\space w})$

We will be using the same example like the last time.

| **Word** | **Document Frequency** |
| --- | --- |
| **I** | 2 |
| **like** | 2 |
| **pizza** | 2 |
| **do** | 1 |
| **not** | 1 |

| **Word** | **Document Frequency** | **IDF** |
| --- | --- | --- |
| **I** | 2 | 0 |
| **like** | 2 | 0 |
| **pizza** | 2 | 0 |
| **do** | 1 | 0.693 |
| **not** | 1 | 0.693 |

*Now let’s calculate the IDFs :*

$log(\frac{2}{2})=0 \space |\space log(\frac{2}{1})=0.693$

IDF helps downweight common words (like *"I"*) and emphasize rare words (like *"do"* and *"not"*), which are often more meaningful for distinguishing between documents.

### `TF-IDF`

TF tells us how important a word is in a document, IDF tells us how rare that word is. Combining them gives a score that tells us how important a word is in a document, while leaving behind common words.

*The formula is :*

$TF-IDF(w)=TF(w)\times IDF(w)$

We will be using the same example like the last time.

TF-IDF highlights that words like *"do"* and *"not"* are more important to Sentence 2 because they are rare in the dataset.

| **Word** | **TF-IDF in S1** | **TF-IDF in S2** |
| --- | --- | --- |
| **I** | 0.33 x 0 = 0 | 0.2 x 0 = 0 |
| **like** | 0.33 x 0 = 0 | 0.2 x 0 = 0 |
| **pizza** | 0.33 x 0 = 0 | 0.2 x 0 = 0 |
| **do** | 0 x 0.69 = 0 | 0.2 x 0.69 = 0.13 |
| **not** | 0 x 0.69 = 0 | 0.2 x 0.69 = 0.13 |

---

# `Word Embedding`

## `Introduction`

Word embeddings involves representing words as numerical vectors in a continuous vector space. Here, each word is mapped to a vector of real numbers in a high-dimensional space.

They help us to understand how words are related to each other, and hence we can solve questions like

$King-Man+Woman\sim Queen$

They capture the contextual meaning behind all the words, hence understanding a language in depth.

## `Word2Vec`

`*Original Paper : https://arxiv.org/pdf/1301.3781*`

What word2vec works in such a way, that it causes words that tend to occur in same type of contexts, to have embedding values similar to each other.

Let’s consider two sentences.

S1 : `The child threw the ball across the park.`

S2 : `The kid threw the ball across the park.`

Here, due to word2vec, the words `child` and `kid` will have similar embedding vectors, as they have similar context in these sentences.

- How does it work?
    
    ![image.png](Basics%20of%20NLP%2014c17bf3de1e80a5bbeaf3a79696a82d/image%201.png)
    

This is how the Word2Vec model looks like, confusing at first right?
Let’s decode its parts one by one.

### `CBOW(Continuous Bag of Words)`

Let’s define a vocabulary first

Vocab : `What is life if not a rise to the top`

We create a One Hot Vector for each of the words.

- What is One Hot Vector? (extra)
    
    One position is set to 1, and rest are kept as 0 in a vector Suppose we have three fruit categories: **Apple**, **Banana**, and **Cherry**. We can represent them as one-hot vectors like this:
    
    - **Apple**: [1, 0, 0]
    - **Banana**: [0, 1, 0]
    - **Cherry**: [0, 0, 1]

We need to define a context window now, let it be 3 words.

What we aim to do in CBOW is, we will input the left and right words, which will try to predict the middle most word. In our case, let’s see What is life.

We devise a neural network structure which inputs the two one-hot vectors and converts them to a 3x1 hidden layer, which gets converted into a 5x1 matrix to give the output of the `is` ,after we run it through the activation layer of **Softmax**. Weights are updated throughout all the context windows, `is life if`, then `life if not` and so on, and this how we work with CBOW!

![image.png](Basics%20of%20NLP%2014c17bf3de1e80a5bbeaf3a79696a82d/image%202.png)

### `Skipgram`

Skipgram works exactly in the reverse manner of how CBOW works.

Here we take the centre word and then try to predict the other words surrounding it using a neural network. (I will skip discussion as the process is similar)

![image.png](Basics%20of%20NLP%2014c17bf3de1e80a5bbeaf3a79696a82d/image%203.png)

### `Why use it?`

- It preserves the relationship between word.
- It also deals with new words, by assigning them new vectors.

## `GloVe`

`*Original Paper : https://nlp.stanford.edu/pubs/glove.pdf*`

It is an unsupervised learning algorithm that is designed to learn word embeddings, by making statistical relations between words in a large corpus. It tries to find the co-occurrence patterns in a corpus

Let us consider a set of sentences

1. I enjoy flying.
2. I like NLP.
3. I like deep learning.

From these sentences, let’s list the unique words: `[I, enjoy, flying, like, NLP, deep, learning]`

We need to fix a context window, here too like Word2Vec. Let’s make it 1. That means we will only check co-occurrence with the adjacent word, aka the word earlier and the word later

Now we want to make a co-occurrence matrix X

![image.png](Basics%20of%20NLP%2014c17bf3de1e80a5bbeaf3a79696a82d/image%204.png)

Now let’s understand by taking an example.

See the word `deep` , deep has the words `like` before it and `learning` after it. So logically deep should have only like and learning values set to 1. If you check that is the case horizontally as well as vertically

> This co-occurrence matrix comes out to be symmetric, if you notice. Can you think why is it the case? (Thought exercise for the reader)
> 

Any element in this matrix is denoted by $x_{ij}$ where it means how many times is i succeeded by j in the corpus.

$x_i$  will denote each row, which is basically how many times i with being succeeded by each word in the corpus.   

$x_i=\sum_k{x_{ik}}{}$

Hence, the probability comes out to be:

$$
\color{red} P_{ij}=P(\frac{w_j}{w_i})=\frac{x_{ij}}{x_i}
$$

---

