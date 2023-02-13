
filename=thesis
# Path to output folder
# Video size (by default Full HD)
totalWidth=1920
totalHeight=1080

maxPageNum=60

# The width/height ratio of A4 paper
ratioA4="0.7070"
# The height of an individual tile
tileHeight=`echo "scale=10;sqrt(($totalWidth*$totalHeight)/($ratioA4*$maxPageNum))" | bc`
tileWidth=`echo "scale=10;$tileHeight*$ratioA4" | bc`

# Calculate grid
numTilesHeight=`echo "scale=10;$totalHeight/$tileHeight" | bc`
numTilesWidth=`echo "scale=10;$totalWidth/$tileWidth" | bc`
# Ceil tiles to integers,
numTilesHeight=`awk -v var="$numTilesHeight" 'BEGIN{var = var < 0 ? int(var) : (int(var) + (var == int(var) ? 0 : 1)); print var}'`
numTilesWidth=`awk -v var="$numTilesWidth" 'BEGIN{var = var < 0 ? int(var) : (int(var) + (var == int(var) ? 0 : 1)); print var}'`
# Report measurements
echo -e "\nMovie measurements:"
echo -e "Number of horizontal tiles: $numTilesWidth"
echo -e "Number of vertical tiles: $numTilesHeight"
# Having ceiled the number of tiles, they exceed the totalWidth and totalHeight.
# So, we also need to recalculate the tileHeight and tileWidth. This step will
# (slightly) change the A4 ratio, but it beats half pages in the video.
tileHeight=`echo "scale=5;$totalHeight/$numTilesHeight" | bc`
tileHeight=`echo $tileHeight |cut -f1 -d"."`
tileWidth=`echo "scale=5;$totalWidth/$numTilesWidth" | bc`
tileWidth=`echo $tileWidth |cut -f1 -d"."`
echo -e "\nStart generating images...\n"
montage -density 300 -flatten thesis.pdf -tile ${numTilesWidth}x${numTilesHeight} -background white -geometry ${tileWidth}x${tileHeight} thesis.png
echo "Image thesis.png generated!"
done