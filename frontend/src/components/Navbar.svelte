<script>
  import {Link} from 'svelte-routing'
</script>
  <nav>
    <div class="nav-wrapper">
      <div class="container">
        <Link to="/"><span class="brand-logo">Movie Recommender</span></Link>
        <ul id="nav-mobile" class="right hide-on-med-and-down">
          <li><Link to="/" class="brand-logo">Home</Link></li>
          <li><Link to="/history" class="brand-logo">History</Link></li>
        </ul>
      </div>
    </div>
  </nav>
        