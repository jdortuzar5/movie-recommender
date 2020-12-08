import App from './App.svelte';

const app = new App({
	target: document.body,
	props: {
		name: 'Domingo'
	}
});

export default app;