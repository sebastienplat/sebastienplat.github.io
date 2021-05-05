# History

#### Introduction

**Natural language processing** (**NLP**) is a field of
[human–computer interaction](https://en.wikipedia.org/wiki/human–computer_interaction) whose goal is to make computers derive meaning from human or natural language input. Many challenges in NLP involve: 

+ [natural language understanding](https://en.wikipedia.org/wiki/natural_language_understanding)
+ [natural language generation](https://en.wikipedia.org/wiki/natural_language_generation)


#### 1950s to 1970s

In 1950, Alan Turing published an article which proposed what is now called the [Turing test](https://en.wikipedia.org/wiki/Turing_test) as a criterion of intelligence.

In 1954, the [Georgetown experiment](https://en.wikipedia.org/wiki/Georgetown-IBM_experiment) provided the fully automatic translation of more than sixty Russian sentences into English. 
The authors claimed that within three or five years, machine translation would be a solved problem.

Some successful NLP systems were developed in the 1960s, like [SHRDLU](https://en.wikipedia.org/wiki/SHRDLU), a natural language system working in restricted "[blocks worlds](https://en.wikipedia.org/wiki/blocks_world)" with restricted vocabularies, and 
[ELIZA](https://en.wikipedia.org/wiki/ELIZA), a simulation of a [Rogerian psychotherapist](https://en.wikipedia.org/wiki/Rogerian_psychotherapy). 
Using almost no information about human thought or emotion, ELIZA sometimes provided a startlingly human-like interaction, but with generic responses when the "patient" exceeded the very small knowledge base ("My head hurts" would be answered by "Why do you say your head hurts?").

Many programmers began to write 'conceptual ontologies' in the 1970s, which structured real-world information into computer-understandable data. During this time, many chatterbots] were also written, including 
[PARRY](https://en.wikipedia.org/wiki/PARRY),
[Racter](https://en.wikipedia.org/wiki/Racter), and
[Jabberwacky](https://en.wikipedia.org/wiki/Jabberwacky). 

These NLP systems were often based on complex sets of hand-written rules. 


#### 1980s onward

A major breakthrough occured in the late 1980s, with the introduction of machine learning algorithms based on [corpus linguistics](https://en.wikipedia.org/wiki/corpus_linguistics), the systematic investigation of large [textual corpora](https://en.wikipedia.org/wiki/text_corpus) to identify real-world patterns. These algorithms, made possible by the increase in computational power, were used to create the first 
[statistical machine translation](https://en.wikipedia.org/wiki/statistical_machine_translation) systems. 

The dominant theories of linguistics at the time discouraged the application of such models, as they were supposedly not efficient for language processing and its many corner cases
(see Chomskyan linguistics,
[transformational grammar](https://en.wikipedia.org/wiki/transformational_grammar) and
[poverty of the stimulus](https://en.wikipedia.org/wiki/poverty_of_the_stimulus)).

Early successes at IBM Research used existing multilingual corpora produced by the Parliament of Canada and the European Union. Later research has improved the efficiency of learning from limited amounts of data. Recent research focus on [unsupervised](https://en.wikipedia.org/wiki/unsupervised_learning) and [semi-supervised learning](https://en.wikipedia.org/wiki/semi-supervised_learning) algorithms, which learn from data that has not been hand-annotated with the desired answers. 

This task is generally much more difficult than supervised learning, and typically produces less accurate results for a given amount of input data. However, there is an enormous amount of non-annotated data available (including, among other things, the entire content of the Web), which can often make up for the inferior results.

#### Future

NLP research is gradually shifting from lexical semantics to compositional semantics and, further on, narrative understanding. Human-level natural language processing, however, is an
[AI-complete](https://en.wikipedia.org/wiki/AI-complete) problem. That is, it is equivalent to solving the central artificial intelligence problem—making computers as intelligent as people, or
[strong AI](https://en.wikipedia.org/wiki/artificial_general_intelligence). NLP's future is therefore tied closely to the development of AI in general.

#### Standardization

An ISO subcommittee is working in order to ease interoperability between
[lexical resources](https://en.wikipedia.org/wiki/lexical_resource) and NLP programs. The
subcommittee is part of [ISO/TC37](https://en.wikipedia.org/wiki/ISO/TC37) and is called
ISO/TC37/SC4. Some ISO standards are already published but most of them
are under construction, mainly on lexicon representation (see
[LMF](https://en.wikipedia.org/wiki/lexical_markup_framework)), annotation, and data
category registry.

#### See also

-   [List of natural language processing toolkits](List_of_natural_language_processing_toolkits)
-   [Biomedical text mining](https://en.wikipedia.org/wiki/Biomedical_text_mining)
-   [Compound term processing](https://en.wikipedia.org/wiki/Compound_term_processing)
-   [Computer-assisted reviewing](https://en.wikipedia.org/wiki/Computer-assisted_reviewing)
-   [Controlled natural language](https://en.wikipedia.org/wiki/Controlled_natural_language)
-   [Deep Linguistic Processing](https://en.wikipedia.org/wiki/Deep_Linguistic_Processing)
-   [Foreign language reading aid](https://en.wikipedia.org/wiki/Foreign_language_reading_aid)
-   [Foreign language writing aid](https://en.wikipedia.org/wiki/Foreign_language_writing_aid)
-   [Language technology](https://en.wikipedia.org/wiki/Language_technology)
-   [Latent semantic indexing](https://en.wikipedia.org/wiki/Latent_semantic_indexing)
-   [LRE Map](https://en.wikipedia.org/wiki/LRE_Map)
-   [Natural language programming](https://en.wikipedia.org/wiki/Natural_language_programming)
-   [Reification (linguistics)](https://en.wikipedia.org/wiki/Reification_(linguistics%29)
-   [Spoken dialogue system](https://en.wikipedia.org/wiki/Spoken_dialogue_system)
-   [Telligent Systems](https://en.wikipedia.org/wiki/Telligent_Systems)
-   [Transderivational search](https://en.wikipedia.org/wiki/Transderivational_search)


# NLP Evaluation

NLP evaluation aims to determine if the algorithm anwsers the goals of its designers and/or meets the users needs. There are several main types of evaluations:

#### Intrinsic v. extrinsic

**Intrinsic evaluation** considers an isolated NLP system and characterizes its performance with respect to a *gold standard* result as defined by the evaluators. 
**Extrinsic evaluation**, also called *evaluation in use,* considers the NLP system in a more complex setting as either an embedded system or a precise function for a human user. The extrinsic performance of the system is then characterized in terms of utility with respect to the overall task of the extraneous system or the human user. 

For example, consider a syntactic parser which is based on the output of some part of speech (POS) tagger. An intrinsic evaluation would run the POS tagger on [structured data](https://en.wikipedia.org/wiki/Data_model), and compare the system output of the POS tagger to the gold standard output. An extrinsic evaluation would run the parser with some other POS tagger, and then with the novel POS tagger, and compare the parsing accuracy.

#### Black-box v. glass-box

**Black-box evaluation** requires someone to run an NLP system on a sample data set and to measure a number of parameters related to: the quality of the process, such as speed, reliability, resource consumption; and most importantly, the quality of the result, such as the accuracy of data annotation or the fidelity of a translation. 
**Glass-box evaluation** looks at the: design of the system; the algorithms that are implemented; the linguistic resources it uses, like vocabulary size or expression set [cardinality](https://en.wikipedia.org/wiki/cardinality). Given the complexity of NLP problems, it is often difficult to predict performance only on the basis of glass-box evaluation; but this type of evaluation is more informative with respect to error analysis or future developments of a system.

#### Automatic v. manual

In many cases, **automatic procedures** can be defined to evaluate an NLP system by comparing its output with the gold standard one. Although the cost of reproducing the gold standard can be quite high, [bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics%29) automatic evaluation on the same input data can be repeated as often as needed without inordinate additional costs. 
However for many NLP problems the precise definition of a gold standard is a complex task and it can prove impossible when inter-annotator agreement is insufficient. 
**Manual evaluation** is best performed by human judges instructed to estimate the quality of a system, or most often of a sample of its output, based on a number of criteria. Although, thanks to their linguistic competence, human judges can be considered as the reference for a number of language processing tasks, there is also considerable variation across their ratings. That is why automatic evaluation is sometimes referred to as *objective* evaluation while the human evaluation is *perspective*.


# Major Tasks in NLP

#### Fundamental Tasks

##### [Sentence breaking](Sentence_breaking "wikilink")
(also known as [sentence boundary disambiguation](sentence_boundary_disambiguation "wikilink")): Given a chunk of text, find the sentence boundaries. Sentence boundaries are often marked by [periods](Full_stop "wikilink") or other [punctuation marks](punctuation_mark "wikilink"), but these same characters can serve other purposes (e.g. marking [abbreviations](abbreviation "wikilink")).

##### [Parsing](Parsing "wikilink")
Determine the [parse tree](parse_tree "wikilink") (grammatical analysis) of a given sentence. The [grammar](grammar "wikilink") for [natural languages](natural_language "wikilink") is [ambiguous](ambiguous "wikilink") and typical sentences have multiple possible analyses. In fact, perhaps surprisingly, for a typical sentence there may be thousands of potential parses (most of which will seem completely nonsensical to a human).

##### [Part-of-speech tagging](Part-of-speech_tagging "wikilink")
Given a sentence, determine the [part of speech](part_of_speech "wikilink") for each word. Many words, especially common ones, can serve as multiple [parts of speech](parts_of_speech "wikilink"). For example, "book" can be a [noun](noun "wikilink") ("the book on the table") or [verb](verb "wikilink") ("to book a flight"); "set" can be a [noun](noun "wikilink"), [verb](verb "wikilink") or [adjective](adjective "wikilink"); and "out" can be any of at least five different parts of speech. Some languages have more such ambiguity than others. Languages with little [inflectional morphology](inflectional_morphology "wikilink"), such as [English](English_language "wikilink") are particularly prone to such ambiguity. [Chinese](Chinese_language "wikilink") is prone to such ambiguity because it is a [tonal language](tonal_language "wikilink") during verbalization. Such inflection is not readily conveyed via the entities employed within the orthography to convey intended meaning.

##### [Word segmentation](Word_segmentation "wikilink")
Separate a chunk of continuous text into separate words. For a language like [English](English_language "wikilink"), this is fairly trivial, since words are usually separated by spaces. However, some written languages like [Chinese](Chinese_language "wikilink"), [Japanese](Japanese_language "wikilink") and [Thai](Thai_language "wikilink") do not mark word boundaries in such a fashion, and in those languages text segmentation is a significant task requiring knowledge of the [vocabulary](vocabulary "wikilink") and [morphology](Morphology_(linguistics) "wikilink") of words in the language.

##### [Word sense disambiguation](Word_sense_disambiguation "wikilink")
Many words have more than one [meaning](Meaning_(linguistics) "wikilink"); we have to select the meaning which makes the most sense in context. For this problem, we are typically given a list of words and associated word senses, e.g. from a dictionary or from an online resource such as [WordNet](WordNet "wikilink").


#### Advanced tasks

##### [Automatic summarization](https://en.wikipedia.org/wiki/Automatic_summarization)
Produce a readable summary of a chunk of text. Often used to provide summaries of text of a known type, such as articles in the financial section of a newspaper.


##### [Coreference resolution](https://en.wikipedia.org/wiki/Coreference)
Given a sentence or larger chunk of text, determine which words ("mentions") refer to the same objects ("entities"). [Anaphora resolution](https://en.wikipedia.org/wiki/Anaphora_resolution) is a specific example of this task, and is specifically concerned with matching up pronouns with the nouns or names that they refer to. The more general task of coreference resolution also includes identifying so-called "bridging relationships" involving referring expressions. 

For example, in a sentence such as "He entered John's house through the front door", "the front door" is a referring expression and the bridging relationship to be identified is the fact that the door being referred to is the front door of John's house (rather than of some other structure that might also be referred to).

##### [Discourse analysis](https://en.wikipedia.org/wiki/Discourse_analysis)
This rubric includes a number of related tasks. One task is identifying the discourse structure of connected text, i.e. the nature of the discourse relationships between sentences (e.g. elaboration, explanation, contrast). Another possible task is recognizing and classifying the speech acts in a chunk of text (e.g. yes-no question, content question, statement, assertion, etc.).

##### [Machine translation](https://en.wikipedia.org/wiki/Machine_translation)
Automatically translate text from one human language to another. This is one of the most difficult problems, and is a member of a class of problems colloquially termed "[AI-complete](https://en.wikipedia.org/wiki/AI-complete)", i.e. requiring all of the different types of knowledge that humans possess (grammar, semantics, facts about the real world, etc.) in order to solve properly.

##### [Morphological segmentation](https://en.wikipedia.org/wiki/Morphology_(linguistics&29)
Separate words into individual [morphemes](morpheme "wikilink") and identify the class of the morphemes. The difficulty of this task depends greatly on the complexity of the [morphology](Morphology_(linguistics) "wikilink") (i.e. the structure of words) of the language being considered. [English](English_language "wikilink") has fairly simple morphology, especially [inflectional morphology](inflectional_morphology "wikilink"), and thus it is often possible to ignore this task entirely and simply model all possible forms of a word (e.g. "open, opens, opened, opening") as separate words. In languages such as [Turkish](Turkish_language "wikilink") or [Manipuri](Manipuri_language "wikilink"),[^4] a highly agglutinated Indian language, however, such an approach is not possible, as each dictionary entry has thousands of possible word forms.

##### [Named entity recognition](Named_entity_recognition "wikilink") (NER)
Given a stream of text, determine which items in the text map to proper names, such as people or places, and what the type of each such name is (e.g. person, location, organization). Note that, although [capitalization](capitalization "wikilink") can aid in recognizing named entities in languages such as English, this information cannot aid in determining the type of named entity, and in any case is often inaccurate or insufficient. For example, the first word of a sentence is also capitalized, and named entities often span several words, only some of which are capitalized. Furthermore, many other languages in non-Western scripts (e.g. [Chinese](Chinese_language "wikilink") or [Arabic](Arabic_language "wikilink")) do not have any capitalization at all, and even languages with capitalization may not consistently use it to distinguish names. For example, [German](German_language "wikilink") capitalizes all [nouns](noun "wikilink"), regardless of whether they refer to names, and [French](French_language "wikilink") and [Spanish](Spanish_language "wikilink") do not capitalize names that serve as [adjectives](adjective "wikilink").

##### [Natural language generation](Natural_language_generation "wikilink")
Convert information from computer databases into readable human language.

##### [Natural language understanding](Natural_language_understanding "wikilink")
Convert chunks of text into more formal representations such as [first-order logic](first-order_logic "wikilink") structures that are easier for [computer](computer "wikilink") programs to manipulate. Natural language understanding involves the identification of the intended semantic from the multiple possible semantics which can be derived from a natural language expression which usually takes the form of organized notations of natural languages concepts. Introduction and creation of language metamodel and ontology are efficient however empirical solutions. An explicit formalization of natural languages semantics without confusions with implicit assumptions such as [closed-world assumption](closed-world_assumption "wikilink") (CWA) vs. [open-world assumption](open-world_assumption "wikilink"), or subjective Yes/No vs. objective True/False is expected for the construction of a basis of semantics formalization.[^5]

##### [Optical character recognition](Optical_character_recognition "wikilink") (OCR)
Given an image representing printed text, determine the corresponding text.

##### [Question answering](Question_answering "wikilink")
Given a human-language question, determine its answer. Typical questions have a specific right answer (such as "What is the capital of Canada?"), but sometimes open-ended questions are also considered (such as "What is the meaning of life?"). Recent works have looked at even more complex questions.[^6]

##### [Relationship extraction](Relationship_extraction "wikilink")
Given a chunk of text, identify the relationships among named entities (e.g. who is married to whom).

##### [Sentiment analysis](Sentiment_analysis "wikilink")
Extract subjective information usually from a set of documents, often using online reviews to determine "polarity" about specific objects. It is especially useful for identifying trends of public opinion in the social media, for the purpose of marketing.

##### [Speech recognition](Speech_recognition "wikilink")
Given a sound clip of a person or people speaking, determine the textual representation of the speech. This is the opposite of [text to speech](text_to_speech "wikilink") and is one of the extremely difficult problems colloquially termed "[AI-complete](AI-complete "wikilink")" (see above). In [natural speech](natural_speech "wikilink") there are hardly any pauses between successive words, and thus [speech segmentation](speech_segmentation "wikilink") is a necessary subtask of speech recognition (see below). Note also that in most spoken languages, the sounds representing successive letters blend into each other in a process termed [coarticulation](coarticulation "wikilink"), so the conversion of the analog signal to discrete characters can be a very difficult process.

##### [Speech segmentation](Speech_segmentation "wikilink")
Given a sound clip of a person or people speaking, separate it into words. A subtask of [speech recognition](speech_recognition "wikilink") and typically grouped with it.

##### [Topic segmentation](Topic_segmentation "wikilink") and recognition
Given a chunk of text, separate it into segments each of which is devoted to a topic, and identify the topic of the segment.

In some cases, sets of related tasks are grouped into subfields of NLP
that are often considered separately from NLP as a whole. Examples
include:

##### [Information retrieval](Information_retrieval "wikilink") (IR)
This is concerned with storing, searching and retrieving information. It is a separate field within computer science (closer to databases), but IR relies on some NLP methods (for example, stemming). Some current research and applications seek to bridge the gap between IR and NLP.

##### [Information extraction](Information_extraction "wikilink") (IE)
This is concerned in general with the extraction of semantic information from text. This covers tasks such as [named entity recognition](named_entity_recognition "wikilink"), [Coreference resolution](Coreference "wikilink"), [relationship extraction](relationship_extraction "wikilink"), etc.

##### [Speech processing](Speech_processing "wikilink")
This covers [speech recognition](speech_recognition "wikilink"), [text-to-speech](text-to-speech "wikilink") and related tasks.

Other tasks include:

-   [Native Language
    Identification](Native_Language_Identification "wikilink")
-   [Stemming](Stemming "wikilink")
-   [Text simplification](Text_simplification "wikilink")
-   [Text-to-speech](Text-to-speech "wikilink")
-   [Text-proofing](Text-proofing "wikilink")
-   [Natural language
    search](Natural_language_user_interface "wikilink")
-   [Query expansion](Query_expansion "wikilink")
-   [Automated essay scoring](Automated_essay_scoring "wikilink")
-   [Truecasing](Truecasing "wikilink")

# Principles

Reducing the error rate for an application often involves two antagonist efforts:

+ increasing accuracy or precision (minimizing false positives - type I)
+ increasing coverage or recall (minimizing false negatives - type II)



### NLP using machine learning

Some of the earliest-used machine learning algorithms, such as 
[decision trees](https://en.wikipedia.org/wiki/decision_tree), produced systems of hard if-then rules similar to existing hand-written rules.

[Part-of-speech tagging](https://en.wikipedia.org/wiki/Part_of_speech_tagging) introduced the use of 
[Hidden Markov Models](https://en.wikipedia.org/wiki/Hidden_Markov_Models) to NLP.

Research has increasingly focused on 
[statistical models](https://en.wikipedia.org/wiki/statistical_natural_language_processing), which make soft, probabilistic decisions based on attaching
[real-valued](https://en.wikipedia.org/wiki/real-valued) weights to the features making up the input data — large corpora of typical real-world examples. The 
[cache language models](https://en.wikipedia.org/wiki/cache_language_model), upon which many 
speech recognition systems now rely, are examples of such statistical models. 

Systems based on machine-learning algorithms have many advantages over hand-produced rules:

-   Automatically focus on the most common cases, based on real-world data
- 	Use statistical inference to handle unfamiliar/erroneous input 
-	Can easily become more accurate by supplying more input data

### Statistical NLP

Statistical natural-language processing uses
[stochastic](stochastic "wikilink"),
[probabilistic](probabilistic "wikilink"), and
[statistical](statistical "wikilink") methods to resolve some of the
difficulties discussed above, especially those which arise because
longer sentences are highly ambiguous when processed with realistic
grammars, yielding thousands or millions of possible analyses. Methods
for disambiguation often involve the use of
[corpora](corpus_linguistics "wikilink") and [Markov
models](Markov_model "wikilink"). The ESPRIT Project P26 (1984 - 1988),
led by [CSELT](CSELT "wikilink"), explored the problem of speech
recognition comparing knowledge-based approach and statistical ones: the
chosen result was a completely statistical model.[^7] One among the
first models of statistical natural language understanding was
introduced in 1991 by [Roberto
Pieraccini](Roberto_Pieraccini "wikilink"), Esther Levin, and Chin-Hui
Lee from [Bell Laboratories](Bell_Laboratories "wikilink").[^8] NLP
comprises all quantitative approaches to automated [language
processing](language_processing "wikilink"), including probabilistic
modeling, [information theory](information_theory "wikilink"), and
[linear algebra](linear_algebra "wikilink").[^9] The technology for
statistical NLP comes mainly from [machine
learning](machine_learning "wikilink") and [data
mining](data_mining "wikilink"), both of which are fields of [artificial
intelligence](artificial_intelligence "wikilink") that involve learning
from data.



# Text Processing

#### Corpus Linguistics

##### [Sentence breaking](https://en.wikipedia.org/wiki/Sentence_breaking)
(also known as [sentence boundary disambiguation](https://en.wikipedia.org/wiki/sentence_boundary_disambiguation)): Given a chunk of text, find the sentence boundaries. Sentence boundaries are often marked by periods or other punctuation marks, but these same characters can serve other purposes (e.g. marking abbreviations).

##### [Word segmentation](https://en.wikipedia.org/wiki/Word_segmentation)
Separate a chunk of continuous text into separate words. This is fairly trivial for languages like English, where words are usually separated by spaces. But for languages that do not mark word boundaries (like Chinese, Japanese and Thai), this requires knowledge of the vocabulary and [morphology](https://en.wikipedia.org/wiki/Morphology_(linguistics&29) of words.

##### Word Spelling Correction
Minimum edit distance


##### [Word sense disambiguation](https://en.wikipedia.org/wiki/Word_sense_disambiguation)
Many words have more than one meaning; we have to select the meaning which makes the most sense in context. For this problem, we are typically given a list of words and associated word senses, e.g. from a dictionary or from an online resource such as [WordNet](https://en.wikipedia.org/wiki/WordNet).

##### [Part-of-speech tagging](https://en.wikipedia.org/wiki/Part-of-speech_tagging)
Given a sentence, determine the [part of speech](https://en.wikipedia.org/wiki/part_of_speech) for each word. Many words, especially common ones, can serve as multiple parts of speech. 

For example, "book" can be a noun ("the book on the table") or verb ("to book a flight"); "set" can be a noun, verb or adjective; and "out" can be any of at least five different parts of speech. 

Some languages have more such ambiguity than others, whether because they have little [inflectional morphology](https://en.wikipedia.org/wiki/inflectional_morphology) (like English) or because they are [tonal languages](https://en.wikipedia.org/wiki/tonal_language) (like Chinese).

##### [Parsing](https://en.wikipedia.org/wiki/Parsing)
Determine the [parse tree](https://en.wikipedia.org/wiki/parse_tree) of a given sentence: an ordered, rooted tree that represents its syntactic structure according to some [context-free grammar](https://en.wikipedia.org/wiki/Context-free_grammar). 

The grammar for natural languages is ambiguous: typical sentences have multiple possible analyses, most of which are nonsensical. The development of [treebanks](https://en.wikipedia.org/wiki/Treebank), starting in the early 1990s, has therefore been very beneficial to computational linguistics: large [textual corpora](https://en.wikipedia.org/wiki/text_corpus) that have been first annotated with part-of-speech tags, then completed with syntactic or semantic sentence structures.

#### Markov Models

#### See also

+ [Brown Corpus](https://en.wikipedia.org/wiki/Brown_Corpus) - link includes other largely used corpora
+ [Treebanks](https://en.wikipedia.org/wiki/Treebank) - link includes many treebanks


