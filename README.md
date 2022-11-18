# Does Youtube reflects the overall polarization in the US?
Study of the links between Youtube communities thanks to the comments left by users.

Classification of channels by their shared community from News & Politics main channels


By the Adamantium Forgers

| Name                | Email                       |
|---------------------|-----------------------------|
| Loïc Fischer        | loic.fischer@epfl.ch        |
| Stéphane Weissbaum  | stephane.weissbaum@epfl.ch  |
| Camille Bernelin    | camille.bernelin@epfl.ch    |
| Michel Morales      | michel.morales@epfl.ch      |

Project based on the YouNiverse dataset provided by the EPFL dlab ([Github](https://github.com/epfl-dlab/YouNiverse) and [Dataset](https://zenodo.org/record/4650046#.Y3bCAzOZNpg))


## Abstract
In 2017, with the arrival in power of Donald Trump as president of the United States, the American political world then quickly split between the pro and anti Trump. 

However this trend of a more polarized society goes back even further. According to a [study][1] lead by Jesse M. Shapiro, Brown University, this polarization began in the late 1990s and early 2000s and has been only increasing since, promoted by the detrimental properties of the US voting system that incentivize people to become radical.

Does the same effect can be measure on youtube communities? The goal of this data story would be to analyse the differents ineraction between the communites on youtube. Does they share a common interest?


### Is this political polarization also reflected online?

#### Facebook

According to a [study][2] lead by to brazilan reseacher, the polarization one year after the the 2017 election can be pictures as follows

<img src="./pictures/fb_us_pol.png" alt="fb_us_pol" width="700"/>

*One year after the bitterly divisive election of Donald Trump as U.S. president, American Facebook users on the political right shared virtually no interests with those on the political left. Pablo Ortellado and Marcio Moretto Ribeiro, CC BY*

The polarization of the topics of interest can be cleary identified here.

## Research Questions
In this project, we will try to answer the following questions, by order of priority, in accordance to the time available before the milestone 3:
1. Can we identify communities inside the already classified 56 main US News&Politics channels ? Is there a clear left-right polarization and isolated communities on each side?
2. Can we identify natural communities in the whole dataset, independently of the political orientation?
3. What is the distribution of users polarization? Is it Gaussian? How many very extreme users do we see? For this purpose, we could assign a p-score (polarization or political score) to users : -1 for left, -0.5 for lean left, 0 for center, 0.5 for lean right, 1 for right.
4. Study a few particular «neutral» channels (ex : NASA, MIT, KhanAcademy, ...) or marginal channels (Flat Earth Society) and analyze their audience, thanks to the classified news channels. What percentage of their audience do they share with each political orientation?
5. Study various events by filtering by tags, descriptions or video titles that contains key words, for example "US Election", "Impeachment", "Hunter Biden", "Capitol". Are these events more covered by one political side?


## Additional dataset 

In order to classify the different political orientations of the youtube users, we decided to use the media bias classification given by [Allsides][3]

<img src="./pictures/media_bias_allsides.png" alt="media_bias" width="300" class="center"/>

*US media bias classification* 

As depicted by wikipedia:
> [AllSides][4] is an American company that assesses the political bias of prominent media outlets, and presents different versions of similar news stories from sources of the political right, left, and center, in a mission to show readers news outside their filter bubble. Focusing on online publications, it has rated over 800 sources on a five-point scale: Left, Leans left, Center, Leans right, and Right. Each source is ranked by unpaid volunteer editors, overseen by two staff members holding political biases different from each other. These crowd-sourced reviews are augmented by editorial reviews performed by staff members. Reassessments may be made based on like button results from community feedback. AllSides uses these rankings to produce [media bias][3] charts listing popular sources.

SHOW THAT THE WORK IS ALREADY DONE. GO TECHNICAL (how did we get it, list of files and describe what each of them do, present the final .csv file: what is inside it, what's the size?)


## Methodology

Show that we can handle the data in size
Explain in detail what we filtered
Give mathematical details of the methods used (and libraries)

ADD LIST OF OUR USEFUL FILES

The diffenrent community would be created using the file "youtube_comments". Every user who commented on one of the diffenrent news souces would be assign to a plotical orrientation.



Then a graph linking channels with the nummber of user commenting the two channels as a link would be created. Finally the goal would be to cluster the different channels an alanyse if 




## Proposed timeline
- 18 November: Milestone P2 due
- 25 November: Filter the dataset (eliminate indian channels) and try to answer research questions 1 and 2.
- 02 November: Homework 2 due. No project work
- 09 December: ???
- 16 December: ???
- 23 December: Milestone P3 due. Focus on writing up the report or the data story, preparing the final presentation, formating for the website and the data story, 


## Team organization

- Loïc:
- Camille: 
- Michel : 
- Stéphane: 

[1]: https://www.nber.org/papers/w26669
[2]: https://theconversation.com/mapping-brazils-political-polarization-online-96434
[3]: https://www.allsides.com/media-bias
[4]: https://www.allsides.com
