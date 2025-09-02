def _extract_continuous_segments(df):

    # Step 1: Get index differences
    index_diff = df.index.to_series().diff().fillna(1)

    # Step 2: Identify where the difference is not 1
    breaks = index_diff[index_diff != 1].index

    # Step 3: Split the DataFrame
    sub_dfs = []
    start = 0
    for i in range(1, len(df)):
        if df.index[i] in breaks:
            sub_dfs.append(df.iloc[start:i])
            start = i
    sub_dfs.append(df.iloc[start:])  # Add the last segment
    # Now sub_dfs is a list of DataFrames with continuous indices

    return sub_dfs


def prepare_samples_with_gaps(data, L=720, l=240):
    padded_segments = []
    skipped_segments = []
    by_hand = []
    for df in data:
        # Determine the rows where each property ([GRO, temp, rh, vpd, swp, total_precip, rad]) has gaps.
        channels = {'GRO':          df.loc[df.GRO.isnull()],
                    'temp':         df.loc[df.temp.isnull()], 
                    'rh':           df.loc[df.rh.isnull()], 
                    'vpd':          df.loc[df.vpd.isnull()], 
                    'swp':          df.loc[df.swp.isnull()], 
                    'total_precip': df.loc[df.total_precip.isnull()], 
                    'rad':          df.loc[df.rad.isnull()]
                }
        # Each channel contains all the rows where a particular property has gaps. The channel is actually a dataframe with all the columns shown. However, each channel assures that all the time stamps (rows) with 
        # gaps of a particular property are shown. This does not exclude the fact that there might be other columns in the data frame that also have gaps. 
        for key, channel in channels.items():
            # Each channel contains all the time stamps for which a particular property has gaps. However, this does not mean that all the rows correspond to a single continuous time period. 
            if len(channel) > 0: # Make sure that there are gaps in this channel.
                # It might and probably does have multiple gaps that have to be separated.
                segments = _extract_continuous_segments(channel)
                # 'segments' contains data frames, where each represents a single, continuos chain of gpas in the data set.  
                for segment in segments:
                    if len(segment) < l: # We only consider segments that are less than 10 days in size. 
                        # We want the gap to be in the middle of the predetermined segment size required for the gap-filling model
                        padding_length = int((L - len(segment))/2) # determines the padding that should be to the left and right of the missing data
                        odd = (L - len(segment))%2 # If the gap size is odd, then the total padding is also odd. Therefore, the right padding is one time stamp longer than the right
                        start = segment.index[0] # determines the index at which the gaps start
                        end = segment.index[-1] # determines the index at which teh gaps end

                        start_padding = start-padding_length
                        end_padding = end+padding_length+odd # If the gap size is odd, then the right padding is one time stamp longer than the right
                        # make sure that the start of the padding index actualy exists
                        try:
                            # TODO: It is possible that the segment with gaps under consideration is at the beginning or end of a time series, so that an error is obtained when 'before' or 'after' is being accessed. 
                            #   The beginning or end of the padding might fall outside the range of the array. For this reason the try-except cycle is used. Find a better solution.
                            before = df.loc[start_padding:start-1] # construct the left padding
                            after = df.loc[end+1:end_padding]
                            padded_segment = pd.concat([before, segment, after])
                            # TODO: It is possible that the added padding also has missing values of the phyical property considered. We have to check for this. 
                            #   It might be a problem if a large proportion of the padding also has gaps. It could also be a problem if the values missing inside the padding are at the exremens of the constructed segment.
                            #   These are all issues that could be imporoved.
                            # if len(padded_segment.loc[padded_segment[key].isnull()]) > l:
                            padded_segments.append(padded_segment)
                        except:
                            by_hand.append(segment)
                    else:
                        skipped_segments.append(segment)
    return padded_segments, skipped_segments, by_hand
                