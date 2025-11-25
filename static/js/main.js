document.addEventListener('DOMContentLoaded', () => {
	// Mobile-friendly enhancements
	initMobileOptimizations();
	initFormEnhancements();
	initTouchOptimizations();
});

function initMobileOptimizations() {
	// Prevent zoom on input focus (iOS)
	const inputs = document.querySelectorAll('input, textarea');
	inputs.forEach(input => {
		input.addEventListener('focus', () => {
			if (window.innerWidth < 768) {
				document.querySelector('meta[name="viewport"]').content = 
					'width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no';
			}
		});
		
		input.addEventListener('blur', () => {
			document.querySelector('meta[name="viewport"]').content = 
				'width=device-width, initial-scale=1, maximum-scale=5, user-scalable=yes';
		});
	});
	
	// Smooth scrolling for anchor links
	document.querySelectorAll('a[href^="#"]').forEach(anchor => {
		anchor.addEventListener('click', function (e) {
			e.preventDefault();
			const target = document.querySelector(this.getAttribute('href'));
			if (target) {
				target.scrollIntoView({
					behavior: 'smooth',
					block: 'start'
				});
			}
		});
	});
}

function initFormEnhancements() {
	// Auto-resize textarea
	const textareas = document.querySelectorAll('textarea');
	textareas.forEach(textarea => {
		textarea.addEventListener('input', function() {
			this.style.height = 'auto';
			this.style.height = Math.min(this.scrollHeight, 300) + 'px';
		});
	});
	
	// Form validation feedback
	const forms = document.querySelectorAll('form');
	forms.forEach(form => {
		form.addEventListener('submit', function(e) {
			const requiredFields = form.querySelectorAll('[required]');
			let isValid = true;
			
			requiredFields.forEach(field => {
				if (!field.value.trim()) {
					field.style.borderColor = 'rgba(255, 0, 80, 0.5)';
					isValid = false;
				} else {
					field.style.borderColor = 'rgba(127, 252, 255, 0.25)';
				}
			});
			
			if (!isValid) {
				e.preventDefault();
				// Show error message
				const existingError = form.querySelector('.form-error');
				if (!existingError) {
					const errorDiv = document.createElement('div');
					errorDiv.className = 'alert form-error';
					errorDiv.textContent = 'Please fill in all required fields.';
					form.insertBefore(errorDiv, form.firstChild);
				}
			}
		});
	});
}

function initTouchOptimizations() {
	// Add touch feedback for buttons
	const buttons = document.querySelectorAll('.btn-primary, .btn-secondary, .nav-link');
	buttons.forEach(button => {
		button.addEventListener('touchstart', function() {
			this.style.transform = 'scale(0.98)';
		});
		
		button.addEventListener('touchend', function() {
			this.style.transform = '';
		});
	});
	
	// Prevent double-tap zoom on buttons
	buttons.forEach(button => {
		button.addEventListener('touchend', function(e) {
			e.preventDefault();
			this.click();
		});
	});
	
	// Optimize for touch devices
	if ('ontouchstart' in window) {
		document.body.classList.add('touch-device');
		
		// Increase tap targets for touch devices
		const smallButtons = document.querySelectorAll('.nav-link');
		smallButtons.forEach(button => {
			button.style.minHeight = '44px';
			button.style.minWidth = '44px';
		});
	}
}

// Handle orientation changes
window.addEventListener('orientationchange', () => {
	setTimeout(() => {
		// Recalculate layouts after orientation change
		window.dispatchEvent(new Event('resize'));
	}, 100);
});

// Performance optimization: Debounce resize events
let resizeTimeout;
window.addEventListener('resize', () => {
	clearTimeout(resizeTimeout);
	resizeTimeout = setTimeout(() => {
		// Handle resize-specific optimizations
		const isMobile = window.innerWidth < 768;
		document.body.classList.toggle('mobile-layout', isMobile);
	}, 250);
});
