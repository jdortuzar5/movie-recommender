<script>
	import "../node_modules/materialize-css/dist/css/materialize.min.css"
	import "../node_modules/materialize-css/dist/js/materialize.min.js"
	import { Router, Link, Route } from "svelte-routing";
	import Navbar from "./components/Navbar.svelte";
	import History from "./pages/History.svelte";
	import Home from "./pages/Home.svelte"
	
</script>


<style>
	main {
		text-align: center;
		padding: 1em;
		max-width: 240px;
		margin: 0 auto;
		background-color: #d5274c;
	}

	h1 {
		color: #ff3e00;
		text-transform: uppercase;
		font-size: 4em;
		font-weight: 100;
	}

	@media (min-width: 640px) {
		main {
			max-width: none;
		}
	}
</style>

<Router>
	<Navbar/>
	<div class="container">
		<Route path="/" component={Home} />
		<Route path="/history" component={History} />
	</div>
</Router>