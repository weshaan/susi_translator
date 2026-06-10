disable_toc: true

<h1>Translation Listener</h1>
<div id="header"></div>
  <input type="text" id="tenant_id" placeholder="Tenant ID" value="0000">
  <button id="newsessionBtn">New Session</button>
</div>
<div id="transcript-container"></div>

<script>
  let transcriptContainer = document.getElementById('transcript-container');
  let pollingInProgress = false;  // This flag will ensure serialized requests

  function adjustTranscriptContainerHeight() {
    const headerHeight = document.getElementById('header').offsetHeight;
    const windowHeight = window.innerHeight;
    transcriptContainer.style.height = (windowHeight - headerHeight - 80) + 'px';
  }

  document.getElementById('newsessionBtn').addEventListener('click', newSession);
  
  function newSession() {
    while (transcriptContainer.firstChild) {
      transcriptContainer.removeChild(transcriptContainer.firstChild);
    }
  }

  function getLatestTranscript() {
    if (pollingInProgress) {
      return;  // Skip if a request is already in progress
    }

    pollingInProgress = true;  // Set flag indicating polling has started

    const tenant_id = document.getElementById('tenant_id').value;
    let transcripts_url = `/api/transcripts?tenant_id=${tenant_id}`;

    fetch(transcripts_url)
      .then(response => response.json())
      .then(data => {
        // GET /api/transcripts returns { "transcripts": [ { chunk_id, transcript }, ... ] }.
        // Iterate over the array, creating/updating one div per chunk_id.
        const transcripts = data.transcripts || [];
        for (let i = 0; i < transcripts.length; i++) {
          const chunk_id = String(transcripts[i].chunk_id);
          const transcript = transcripts[i].transcript;

          // find the div with the chunk_id
          let div = document.getElementById(chunk_id);
          if (div === null) {
            // New chunk ID, add a new line to the transcript container
            const newLine = document.createElement('div');
            newLine.id = chunk_id;
            newLine.textContent = transcript;
            transcriptContainer.appendChild(newLine);
          } else {
            // Same chunk ID, update the existing transcript
            div.textContent = transcript;
          }
        }

        // Scroll to the bottom of the transcript container
        transcriptContainer.scrollTop = transcriptContainer.scrollHeight;
      })
      .catch(error => console.error('Error fetching transcripts:', error))
      .finally(() => {
        pollingInProgress = false;  // Reset the flag once the polling is done

        // Schedule the next poll after this one is done
        setTimeout(getLatestTranscript, 1000);  // Poll every 1 second
      });
  }

  // Adjust transcript container height on load and resize
  window.addEventListener('load', () => {
    adjustTranscriptContainerHeight();
    getLatestTranscript();  // Start polling when the page loads
  });
  
  window.addEventListener('resize', adjustTranscriptContainerHeight);
</script>
