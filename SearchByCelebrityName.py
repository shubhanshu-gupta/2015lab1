
# coding: utf-8

# In[23]:

import urllib2
import urlparse
from BeautifulSoup import BeautifulSoup
from mechanize import Browser
import re

def getunicode(soup):
    body=''
    if isinstance(soup, unicode):
        soup = soup.replace('&#39;',"'")
        soup = soup.replace('&quot;','"')
        soup = soup.replace('&nbsp;',' ')
        body = body + soup
    else:
        if not soup.contents:
            return ''
        con_list = soup.contents
        for con in con_list:
            body = body + getunicode(con)
    return body


def main():
    celeb = str(raw_input('Celebrity Name: '))
    celeb_search = '+'.join(celeb.split())
    
    base_url = 'http://www.imdb.com/find?q='
    url = base_url+celeb_search+'&s=all'
    
    celebrity_search = re.compile('/name/nm\d+')
    
    br = Browser()

    
                     
    br.open(url)

    link = br.find_link(url_regex = re.compile(r'/name/nm.*'))
    res = br.follow_link(link)
    
    soup = BeautifulSoup(res.read())
    
    knownfor=[]
    info = soup.find('div',{'id':'knownfor'})
    r1 = info.find('',{'title':True})['title']
    knownforlist = info.findAll('a',{'href':True})
    
    for i in range(len(knownforlist)):
        knownfor.append(getunicode(knownforlist[i]))
 
    print 'known for: '
    print ', '.join(knownfor)
#    for a in info.findAll('a',{'href':True})[1::2]:
#        print a['href']
    
    movie_url=[]
    knownmovie=[]
    movie_base_url = 'http://www.imdb.com/'
    
    for a in info.findAll('a',{'href':True})[1::2]:
        movie_url.append(a['href'])
        
    for j in range(len(movie_url)):    
        join = urlparse.urljoin(movie_base_url,movie_url[j])
        knownmovie.append(join.encode("UTF-8"))
    
    
#    print knownmovie
    print knownmovie
    genre=[]
    
    for k in range(len(knownmovie)):
        response = urllib2.urlopen(knownmovie[k])
        html = response.read()
        soup1 = BeautifulSoup(html)        
        rate = soup1.find('span',itemprop='ratingValue')
        rating = getunicode(rate)
        des = soup1.find('meta',{'name':'description'})['content']
        infobar = soup1.find('div',{'class':'infobar'})
        r = infobar.find('',{'title':True})['title']
        genrelist = infobar.findAll('a',{'href':True})
        for l in range(len(genrelist)-1):
            genre.append(getunicode(genrelist[l]))
        release_date = getunicode(genrelist[-1])
        review = soup1.find('div',{'class':'user-comments'})
        rev = soup1.find('p', itemprop='reviewBody')
       # print movie_title,rating+'/10.0'
        print 'Release Date:',release_date
        print 'Rated',r
        print ''
        print 'Genre:',
        print ', '.join(genre)
        print '\nDescription:'
        print des
        print '\nReview:'
        print getunicode(rev)
        print ''
        print ''
    
            
#    movie_title = getunicode(soup.find('title'))
#    rate = soup.find('span',itemprop='ratingValue')
#    rating = getunicode(rate)
    
#    actors=[]
#    actors_soup = soup.findAll('a',itemprop='actors')
#    for i in range(len(actors_soup)):
#        actors.append(getunicode(actors_soup[i]))
    
#    des = soup.find('meta',{'name':'description'})['content']

#    genre=[]
#    infobar = soup.find('div',{'class':'infobar'})
#    r = infobar.find('',{'title':True})['title']
#    genrelist = infobar.findAll('a',{'href':True})
    
#    for i in range(len(genrelist)-1):
#        genre.append(getunicode(genrelist[i]))
#    release_date = getunicode(genrelist[-1])

#    review = soup.find('div',{'class': 'user-comments'})
#    rev = soup.find('p', itemprop='reviewBody')
    
#    print movie_title,rating+'/10.0'
#    print 'Relase Date:',release_date
#    print 'Rated',r
#    print ''
#    print 'Genre:',
#    print ', '.join(genre)
#    print '\nActors:',
#    print ', '.join(actors)
#    print '\nDescription:'
#    print des    
#    print '\nReview:'
#    print rev
#    print r1
#    print 'knownfor:',
#    print ', '.join(knownfor)
    
if __name__ == '__main__':
    main()


# In[ ]:



