<script>
    import { onMount } from "svelte";
    let response = [];
    export let movieTitles;
    onMount(async () => {
        const response = await fetch("http://localhost:8000/autocomplete")
        var options = { data:  await response.json(),
                        limit: 5};
        var elems = document.querySelectorAll(".autocomplete");
        var instances = M.Autocomplete.init(elems, options);
    });
    
    function addNewMovie(){
      var newMovie = document.getElementById("autocomplete-input").value
      movieTitles = [...movieTitles, newMovie]
    };
    

</script>

<style>

</style>

<div class="container">
  <div class="row">
    <div class="col s12">
      <div class="row">
        <div class="input-field col s12">
          <input type="text" id="autocomplete-input" class="autocomplete" />
          <label for="autocomplete-input">Movies</label>
          <button class="btn waves-effect waves-light" type="submit" name="action" on:click={() => addNewMovie()}>Add
            <i class="material-icons right">Movie</i>
          </button>
        </div>
      </div>
    </div>
  </div>
</div>
