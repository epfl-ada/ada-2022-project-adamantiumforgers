# ada-2022-project-adamantiumforgers

By the Adamantium Forgers

| Name                | Email                       |
|---------------------|-----------------------------|
| Loïc Fischer        | loic.fischer@epfl.ch        |
| Stéphane Weissbaum  | stephane.weissbaum@epfl.ch  |
| Camille Bernelin    | camille.bernelin@epfl.ch    |
| Michel Morales      | michel.morales@epfl.ch      |

Dataset Source: https://github.com/epfl-dlab/YouNiverse

# Does Youtube reflects the overall polarization in the US?

## Astract
In 2017, with the arrival in power of Donald Trump as president of the United States, the American political world then quickly split between the pro and anti Trump. However this trend of a more polarazed society goes back even further. According to a [study][1] lead by Jesse M. Shapiro, Brown University, this polarization began in the late 1990s and early 2000s and has been only increasing since.

### Is this political polarisation also reflected online?

#### Facebook

According to a [study][2] lead by to brazilan reseacher, the polarization one year after the the 2017 election can be pictures as follows

<img src="./pictures/fb_us_pol.png" alt="fb_us_pol" width="700"/>

*One year after the bitterly divisive election of Donald Trump as U.S. president, American Facebook users on the political right shared virtually no interests with those on the political left. Pablo Ortellado and Marcio Moretto Ribeiro, CC BY*

The polarization of the topics of interest can be cleary identified here.

#### Youtube

But what about youtube. Does the same effect can be measure on youtube communities?
In order to classify the different orientations of the youtube users we decided to used the media bias classification given by [Allsides][3]

<img src="./pictures/media_bias_allsides.png" alt="media_bias" width="300" class="center"/>

*US media bias classification* 

The goal of this data story would be to analyse the differents ineraction between the communites on youtube. Does they share a commun interest?
## Aditional dataset
As depicted by wikipedia:
"[AllSides][4] is an American company that assesses the political bias of prominent media outlets, and presents different versions of similar news stories from sources of the political right, left, and center, in a mission to show readers news outside their filter bubble. Focusing on online publications, it has rated over 800 sources on a five-point scale: Left, Leans left, Center, Leans right, and Right. Each source is ranked by unpaid volunteer editors, overseen by two staff members holding political biases different from each other. These crowd-sourced reviews are augmented by editorial reviews performed by staff members. Reassessments may be made based on like button results from community feedback. AllSides uses these rankings to produce [media bias][3] charts listing popular sources."



## Methodology
The diffenrent community would be created using the file "youtube_comments". Every user who commented on one of the diffenrent news souces would be assign to a plotical orrientation.

Then a graph linking channels with the nummber of user commenting the two channels as a link would be created. Finally the goal would be to cluster the different channels an alanyse if 

[1]: https://www.nber.org/papers/w26669
[2]: https://theconversation.com/mapping-brazils-political-polarization-online-96434
[3]: https://www.allsides.com/media-bias
[4]: https://www.allsides.com
