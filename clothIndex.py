
clothPath = "test_clothes/"
edgePath = "test_edge/"
bgPath = "backgrounds/"
def getCloth(id):
    clothIndex = {
        "0": "003434_1.jpg",\
        "1" : "006026_1.jpg",\
        "2": "010567_1.jpg",\
        "3" : "014396_1.jpg",\
        "4" : "017575_1.jpg",\
        "5" : "019119_1.jpg",\
        "6" : "coat4.png",\
        "7" : "coat5.png",\
        "8" : "coat6.png",\
    }
    
    # print("%s%s"%(clothPath,clothIndex[id]), "%s%s"%(edgePath,clothIndex[id]))
    return "%s%s"%(clothPath,clothIndex[id]), "%s%s"%(edgePath,clothIndex[id])

def getBG(id):
    bgIndex = {
        "0": '0',\
        "1" : "pexels-pixabay-76969.jpg",\
        "2": "pexels-sparsh-karki-2350074.jpg",\
        "3" : "pexels-steve-chai-5204175.jpg",\
        "4" : "pexels-nathan-cowley-1151282.jpg",\
    }
    
    # print("%s%s"%(clothPath,clothIndex[id]), "%s%s"%(edgePath,clothIndex[id]))
    return "%s%s"%(bgPath,bgIndex[id])
