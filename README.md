# studybuddy-api

Base URL: https://studybuddy-api-t716.onrender.com

1. Image Process:
      Url : /studybuddy/process-image
      Request data: form-data
    
      Form field: req -> object
                  req = { 
                    			“message”: string ( user prompt ),
                    			“history”: [] ( Initaly Empty array\list )
                    		}
      		        image: file = Image ( image  uploaded by user )
      Response: {
            		   “reply”: “LLM generate output”,
            		   “history”: [] 
                }
      ** In response history is list of objects where each object contain role & its content, this response formate will be same for every route.
      e.g., [
      		    {
        		    “role”: “human”,
        		    “content”: “Explain what is the AI?”
              },
              {
        		    “role”: “ai”,
        		    “content”: “LLM response”
              }
              
            ]


2. PDF Process:
      Url : /studybuddy/process-pdf
      Request data: form-data
      
      Form field: req -> object
                  req = { 
                    			“message”: string ( user prompt ),
                    			“history”: [] ( Initaly Empty array\list )
      		              }
                  pdf: file = PDF( pdf uploaded by user )
      Response: {
            		   “reply”: “LLM generate output”,
            		   “history”: [] 
                }

3. Audio Process:
    Url : /studybuddy/process-audio
    Request data: form-data
    
    Form field: req -> object
                req = { 
                  			“message”: string ( user prompt ),
                  			“history”: [] ( Initaly Empty array\list )
                       }
    		        audio: file = Audio( audio uploaded by user )
    Response: {
          		   “reply”: “LLM generate output”,
          		   “history”: [] 
              }

4. Roadmap Guide:
    Url : /studybuddy/roadmap
    Request data: raw json object
    JSON object: {
                    “message”: “user input”,
                    “history”: []
                 }
     
    Response: {
          		   “reply”: “LLM generate output”,
          		   “history”: [] 
              }

5. YT-video Process: 
    Url : /studybuddy/roadmap
    Request data: raw json object
    JSON object: {
              			“message”: “user input which contains yt-link”,
              			“history”: []
                 }
     
    Response: {
          		   “reply”: “LLM generate output”,
          		   “history”: [] 
              }
