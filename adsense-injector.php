<?php
/**
 * Plugin Name: AdSense Injector for PedPro
 * Description: Injects Google AdSense script tag into <head> on every page.
 *              Also adds privacy policy if missing.
 * Version: 1.0.0
 * Author: Auto-Blogger-WP
 */

if (!defined('ABSPATH')) exit;

// ── AdSense <head> injection ──────────────────────────────────────────────
// Replace ca-pub-XXXXXXXXXXXXXXXX with your real AdSense publisher ID.
add_action('wp_head', function () {
    $pub_id = defined('ADSENSE_PUB_ID') ? ADSENSE_PUB_ID : 'ca-pub-XXXXXXXXXXXXXXXX';
    ?>
    <!-- Google AdSense -->
    <script async src="https://pagead2.googlesyndication.com/pagead/js/adsbygoogle.js?client=<?php echo esc_attr($pub_id); ?>"
         crossorigin="anonymous"></script>
    <?php
});

// ── Auto-create Privacy Policy page if it doesn't exist ───────────────────
add_action('after_setup_theme', function () {
    // Check if a privacy policy page already exists
    $privacy_page_id = get_option('wp_page_for_privacy_policy');
    if ($privacy_page_id) {
        $page = get_post($privacy_page_id);
        if ($page && $page->post_status === 'publish') {
            return; // Already exists and published
        }
    }

    // Search for any page with slug 'privacy-policy'
    $existing = get_posts([
        'post_type'   => 'page',
        'name'        => 'privacy-policy',
        'post_status' => 'publish',
        'numberposts' => 1,
    ]);
    if (!empty($existing)) {
        update_option('wp_page_for_privacy_policy', $existing[0]->ID);
        return;
    }

    // Create the privacy policy page (EN)
    $en_content = '<!-- wp:heading --><h2>Privacy Policy for PedPro</h2><!-- /wp:heading -->
<!-- wp:paragraph --><p><strong>Last updated:</strong> ' . date('F j, Y') . '</p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>1. Introduction</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>Welcome to PedPro ("we," "our," or "us"). We are committed to protecting your privacy. This Privacy Policy explains how we collect, use, and safeguard your information when you visit our website at <a href="https://pedpro.online">https://pedpro.online</a> (the "Site").</p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>2. Information We Collect</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>We may collect the following types of information:</p><!-- /wp:paragraph -->
<!-- wp:list --><ul><li><strong>Usage Data:</strong> We use Google Analytics and similar tools to collect anonymous usage data including pages visited, time spent on the site, browser type, and device information.</li><li><strong>Cookies:</strong> We use cookies and similar tracking technologies to enhance your browsing experience and serve relevant advertisements.</li><li><strong>Personal Information:</strong> If you contact us or subscribe to our newsletter, we may collect your name and email address.</li></ul><!-- /wp:list -->

<!-- wp:heading --><h3>3. How We Use Your Information</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>We use the collected information to:</p><!-- /wp:paragraph -->
<!-- wp:list --><ul><li>Provide and maintain our website and services</li><li>Display relevant advertisements through Google AdSense</li><li>Analyze website usage to improve content and user experience</li><li>Communicate with you if you have opted in to receive updates</li><li>Comply with legal obligations</li></ul><!-- /wp:list -->

<!-- wp:heading --><h3>4. Advertising</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>We use Google AdSense to display advertisements on our Site. Google AdSense may use cookies and web beacons to serve ads based on your prior visits to our Site or other websites. You can opt out of personalized advertising by visiting <a href="https://www.google.com/settings/ads">Google Ads Settings</a>.</p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>5. Third-Party Links</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>Our Site may contain links to third-party websites, including affiliate links (Shopee). We are not responsible for the privacy practices of these third-party sites. We encourage you to review their privacy policies before providing any personal information.</p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>6. Data Security</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>We implement appropriate technical and organizational measures to protect your information. However, no method of transmission over the Internet is 100% secure, and we cannot guarantee absolute security.</p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>7. Your Rights</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>Depending on your location, you may have the following rights:</p><!-- /wp:paragraph -->
<!-- wp:list --><ul><li>Access, correct, or delete your personal data</li><li>Object to or restrict processing of your data</li><li>Data portability</li><li>Withdraw consent at any time</li></ul><!-- /wp:list -->

<!-- wp:heading --><h3>8. Children\'s Privacy</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>Our Site is not directed at children under 16. We do not knowingly collect personal information from children. If you believe we have collected information from a child, please contact us immediately.</p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>9. Changes to This Policy</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>We may update this Privacy Policy from time to time. We will notify you of any changes by posting the new policy on this page and updating the "Last updated" date.</p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>10. Contact Us</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>If you have any questions about this Privacy Policy, please contact us at:</p><!-- /wp:paragraph -->
<!-- wp:paragraph --><p>Email: <a href="mailto:tripetkk@gmail.com">tripetkk@gmail.com</a><br>
Website: <a href="https://pedpro.online">https://pedpro.online</a></p><!-- /wp:paragraph -->';

    $page_id = wp_insert_post([
        'post_title'   => 'Privacy Policy',
        'post_content' => $en_content,
        'post_status'  => 'publish',
        'post_type'    => 'page',
        'post_name'    => 'privacy-policy',
    ]);

    if ($page_id && !is_wp_error($page_id)) {
        update_option('wp_page_for_privacy_policy', $page_id);
    }

    // Also create Thai version if not exists
    $th_existing = get_posts([
        'post_type'   => 'page',
        'name'        => 'privacy-policy-th',
        'post_status' => 'publish',
        'numberposts' => 1,
    ]);
    if (empty($th_existing)) {
        $th_content = '<!-- wp:heading --><h2>นโยบายความเป็นส่วนตัว — PedPro</h2><!-- /wp:heading -->
<!-- wp:paragraph --><p><strong>ปรับปรุงล่าสุด:</strong> ' . date('j F Y') . '</p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>1. บทนำ</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>ยินดีต้อนรับสู่ PedPro เราให้ความสำคัญกับการคุ้มครองข้อมูลส่วนบุคคลของคุณ นโยบายความเป็นส่วนตัวฉบับนี้อธิบายถึงวิธีการที่เราเก็บรวบรวม ใช้ และปกป้องข้อมูลของคุณเมื่อคุณเข้าใช้งานเว็บไซต์ <a href="https://pedpro.online">https://pedpro.online</a></p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>2. ข้อมูลที่เราเก็บรวบรวม</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>เราอาจเก็บรวบรวมข้อมูลประเภทต่อไปนี้:</p><!-- /wp:paragraph -->
<!-- wp:list --><ul><li><strong>ข้อมูลการใช้งาน:</strong> เราใช้ Google Analytics และเครื่องมือที่คล้ายคลึงกันในการเก็บข้อมูลการใช้งานแบบไม่ระบุตัวตน รวมถึงหน้าที่เข้าชม เวลาที่ใช้บนเว็บไซต์ ประเภทเบราว์เซอร์ และข้อมูลอุปกรณ์</li><li><strong>Cookies:</strong> เราใช้คุกกี้และเทคโนโลยีการติดตามที่คล้ายคลึงกันเพื่อเพิ่มประสบการณ์การท่องเว็บและแสดงโฆษณาที่เกี่ยวข้อง</li><li><strong>ข้อมูลส่วนบุคคล:</strong> หากคุณติดต่อเราหรือสมัครรับจดหมายข่าว เราอาจเก็บรวบรวมชื่อและอีเมลของคุณ</li></ul><!-- /wp:list -->

<!-- wp:heading --><h3>3. การใช้ข้อมูล</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>เราใช้ข้อมูลที่เก็บรวบรวมเพื่อ:</p><!-- /wp:paragraph -->
<!-- wp:list --><ul><li>ให้บริการและดูแลรักษาเว็บไซต์</li><li>แสดงโฆษณาที่เกี่ยวข้องผ่าน Google AdSense</li><li>วิเคราะห์การใช้งานเพื่อปรับปรุงเนื้อหาและประสบการณ์ผู้ใช้</li><li>สื่อสารกับคุณหากคุณได้ให้ความยินยอม</li><li>ปฏิบัติตามข้อกำหนดทางกฎหมาย</li></ul><!-- /wp:list -->

<!-- wp:heading --><h3>4. โฆษณา</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>เราใช้ Google AdSense เพื่อแสดงโฆษณาบนเว็บไซต์ Google AdSense อาจใช้คุกกี้และเว็บบีคอนเพื่อแสดงโฆษณาตามการเข้าชมก่อนหน้าของคุณ คุณสามารถปิดการใช้งานโฆษณาส่วนบุคคลได้ที่ <a href="https://www.google.com/settings/ads">การตั้งค่าโฆษณาของ Google</a></p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>5. ลิงก์ไปยังเว็บไซต์ภายนอก</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>เว็บไซต์ของเราอาจมีลิงก์ไปยังเว็บไซต์ของบุคคลที่สาม รวมถึงลิงก์พันธมิตร (Shopee) เราไม่รับผิดชอบต่อแนวปฏิบัติด้านความเป็นส่วนตัวของเว็บไซต์เหล่านั้น</p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>6. การรักษาความปลอดภัยของข้อมูล</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>เราดำเนินมาตรการทางเทคนิคและองค์กรที่เหมาะสมเพื่อปกป้องข้อมูลของคุณ อย่างไรก็ตาม ไม่มีวิธีการส่งข้อมูลทางอินเทอร์เน็ตใดที่ปลอดภัย 100%</p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>7. สิทธิของคุณ</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>ขึ้นอยู่กับตำแหน่งที่ตั้งของคุณ คุณอาจมีสิทธิดังต่อไปนี้:</p><!-- /wp:paragraph -->
<!-- wp:list --><ul><li>เข้าถึง แก้ไข หรือลบข้อมูลส่วนบุคคลของคุณ</li><li>คัดค้านหรือจำกัดการประมวลผลข้อมูลของคุณ</li><li>การโอนย้ายข้อมูล</li><li>ถอนความยินยอมได้ตลอดเวลา</li></ul><!-- /wp:list -->

<!-- wp:heading --><h3>8. ความเป็นส่วนตัวของเด็ก</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>เว็บไซต์ของเราไม่มีการกำหนดเป้าหมายไปยังเด็กอายุต่ำกว่า 16 ปี เราไม่ได้เก็บรวบรวมข้อมูลส่วนบุคคลจากเด็กโดยเจตนา</p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>9. การเปลี่ยนแปลงนโยบาย</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>เราอาจอัปเดตนโยบายความเป็นส่วนตัวนี้เป็นครั้งคราว เราจะแจ้งให้คุณทราบเกี่ยวกับการเปลี่ยนแปลงใดๆ โดยการโพสต์นโยบายใหม่บนหน้านี้</p><!-- /wp:paragraph -->

<!-- wp:heading --><h3>10. ติดต่อเรา</h3><!-- /wp:heading -->
<!-- wp:paragraph --><p>หากคุณมีคำถามเกี่ยวกับนโยบายความเป็นส่วนตัวนี้ โปรดติดต่อเราที่:</p><!-- /wp:paragraph -->
<!-- wp:paragraph --><p>อีเมล: <a href="mailto:tripetkk@gmail.com">tripetkk@gmail.com</a><br>
เว็บไซต์: <a href="https://pedpro.online">https://pedpro.online</a></p><!-- /wp:paragraph -->';

        wp_insert_post([
            'post_title'   => 'นโยบายความเป็นส่วนตัว',
            'post_content' => $th_content,
            'post_status'  => 'publish',
            'post_type'    => 'page',
            'post_name'    => 'privacy-policy-th',
        ]);
    }
});
