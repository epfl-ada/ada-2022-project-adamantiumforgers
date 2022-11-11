# La liste

- channels.csv : relates channel_id to channel_num (all channels in News&Politics)
- medias.csv : relates channel_id, channel name and channel_num, for AllTimes medias
- display_id to channels.csv : connects, for all News&Politics channels, display_id to channels_num
- authors_to_channels.csv : for all News&Politics channels, 1 line : this author has comented (at least once) this channel (identified by channel_num)
- graph.csv : all graph edges, connecting two channels that have been commented by the same authors. Weight : number of authors who commented both channels
- graph_test.csv : short version of graph.csv, to perform tests with Gephi