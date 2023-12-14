ffmpeg -framerate 15 -i videoResults/%01d.jpg -c:v libx264 -pix_fmt yuv420p $1
