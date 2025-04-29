import React, { useState } from 'react';
import styles from './Header.module.css';
import logo1 from '../../assets/hdlogo.png';

const Header = ({ onContactClick, onServicesClick, onUsecaseClick }) => {
  const [searchVisible, setSearchVisible] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false); // State for mobile menu

  const toggleMenu = () => {
    setMenuOpen(!menuOpen);
  };

  const slowScrollToTop = (e) => {
    e.preventDefault();
    const start = window.pageYOffset;
    const distance = -start;
    const duration = 500;
    let startTime = null;

    const scroll = (currentTime) => {
      if (!startTime) startTime = currentTime;
      const timeElapsed = currentTime - startTime;
      const scrollAmount = easeInOutQuad(timeElapsed, start, distance, duration);
      window.scrollTo(0, scrollAmount);

      if (timeElapsed < duration) {
        requestAnimationFrame(scroll);
      }
    };

    const easeInOutQuad = (t, b, c, d) => {
      t /= d / 2;
      if (t < 1) return (c / 2) * t * t + b;
      t--;
      return (-c / 2) * (t * (t - 2) - 1) + b;
    };

    requestAnimationFrame(scroll);
  };

  const scrollToBottom = (targetId) => {
    const targetElement = document.getElementById('aboutUS');
    if (!targetElement) return;

    targetElement.scrollIntoView({
      behavior: 'smooth', // Enables smooth scroll
      block: 'start' // Align the target at the top of the viewport
    });
  };

  return (
    <header className={styles.header}>
      <div className={styles.logo}>
        <a href="#home-section" onClick={slowScrollToTop}>
          <img src={logo1} alt="IT Consultancy Logo" className={styles.logoImage} />
        </a>
      </div>
              <h1 className="header-title">DocSumAI</h1>
    </header>
  );
};

export default Header;