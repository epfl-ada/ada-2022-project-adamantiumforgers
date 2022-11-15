# ada-2022-project-adamantiumforgers

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

## Methodology
The diffenrent community would be created using the file "youtube_comments". Every user who commented on one of the diffenrent news souces would be assign to a plotical orrientation.

Then a graph linking channels with the nummber of user commenting the two channels as a link would be created. Finally the goal would be to cluster the different channels an alanyse if 

[1]: https://www.nber.org/papers/w26669
[2]: https://theconversation.com/mapping-brazils-political-polarization-online-96434
[3]: https://www.allsides.com/media-bias
