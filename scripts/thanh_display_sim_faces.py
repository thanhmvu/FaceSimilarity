import os

def save_css(dir):
  f = open(dir + "/style.css","w")
  contents = """
	<style >
	table {
	    font-family: arial, sans-serif;
	    border-collapse: collapse;
	    width: 100%;
	}

	td, th {
		white-space: nowrap;
	    border: 1px solid #dddddd;
	    text-align: left;
	    padding: 8px;
	    font-size: 100%;
	}

	tr:nth-child(even) {
	    background-color: #dddddd;
	}
	</style>"""
  f.write(contents)
  f.close()

def save_html(contents, filename):
  f = open(filename,"w")

  display = """
  <!DOCTYPE html>
  <html>
  <head>
    <link rel="stylesheet" href="style.css">
  </head>
  <body>
  """ + contents + "</body>\n</html>\n"

  f.write(display)
  f.close()

def htmlImg(img, borderColor):
  h = 100
  b = 5
  return "<img src=\""+ img + "\" alt=\"Class " + img + "\" style=\"height:"+`h`+"px; border:"+`b`+"px solid "+ borderColor +";\">"

def print_html(simFaces, dataset_dir, outputFile):
  resultTable = """
  <table>
    <tr>
      <th>Image Idx</th>
      <th>Source Face</th>
      <th>Similar Faces</th>
    </tr>
  """

  for i, srcface in enumerate(simFaces):
		resultTable += """
		<tr>
			<th>"""+ str(i) +"""</th>
			<th>"""+ htmlImg(dataset_dir + srcface,"transparent") + "<br>"+ srcface +"""</th>
		"""
		for simface in simFaces[srcface]:
			img = simface[0]
			dist = simface[1]
			resultTable += "<th>" +htmlImg(dataset_dir + img,"transparent") + "<br>"+ img + "<br> Dist: "+ str(dist) +"</th>"
		resultTable += "</tr>"
    
  resultTable += "</table>\n"

  contents = """<h2>Face Similarity</h2> <br> <br>""" +resultTable
	
  save_css(os.path.dirname(outputFile))
  save_html(contents,outputFile)
  
  
def print_txt(simFaces, outputFile):
  f = open(outputFile,"w")
  delimiter = "\t"
  f.write("i0"+delimiter+"i1"+delimiter+"i2"+delimiter+"i3"+delimiter+"i4"+delimiter+"i5"+delimiter+"i6"+delimiter+"s1"+delimiter+"s2"+delimiter+"s3"+delimiter+"s4"+delimiter+"s5"+delimiter+"s6"+"\n")

  for i,srcface in enumerate(simFaces):
		dict = simFaces[srcface]
		names = delimiter.join([x[0] for x in dict])
		scores = delimiter.join([str(x[1]) for x in dict])
		f.write(srcface +delimiter+ names +delimiter+ scores +"\n")
    
  f.close()
