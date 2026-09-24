import * as THREE from "three";

/**
 * Build a premium procedural pair of sunglasses.
 *
 * Pivot conventions:
 *   - Origin (0,0,0) sits at the center of the nasal bridge.
 *   - +X = wearer's right ear
 *   - +Y = up
 *   - +Z = out of the face, toward the camera
 */
export function createSunglasses(): THREE.Group {
  const group = new THREE.Group();
  group.name = "sunglasses";

  // Premium metallic frame material (reflective titanium / dark gunmetal)
  const frameMat = new THREE.MeshPhysicalMaterial({
    color: 0x141416,
    metalness: 0.90,
    roughness: 0.16,
    clearcoat: 1.0,
    clearcoatRoughness: 0.08,
  });

  // Polished gold accents for hinges and nose pad arms
  const goldAccentMat = new THREE.MeshPhysicalMaterial({
    color: 0xd4af37,
    metalness: 0.95,
    roughness: 0.14,
    clearcoat: 1.0,
    clearcoatRoughness: 0.06,
  });

  // Translucent silicone nose pads
  const padMat = new THREE.MeshPhysicalMaterial({
    color: 0xf5f5f5,
    metalness: 0.0,
    roughness: 0.35,
    transparent: true,
    opacity: 0.85,
  });

  // High-specular optical tinted lenses with realistic PBR surface reflections
  const lensMat = new THREE.MeshPhysicalMaterial({
    color: 0x142030,
    metalness: 0.25,
    roughness: 0.03,
    clearcoat: 1.0,
    clearcoatRoughness: 0.02,
    reflectivity: 0.95,
    transparent: true,
    opacity: 0.74,
    side: THREE.DoubleSide,
  });

  // Balanced classic round sunglasses dimensions
  const LENS_R = 0.0225;                     // 22.5 mm lens radius (~45 mm lens diameter)
  const LENS_TUBE = 0.0016;                  // Sleek 1.6 mm metallic rim
  const EYE_GAP = 0.062;                     // 62 mm pupillary distance
  const BRIDGE_W = EYE_GAP - 2 * LENS_R;     // ~17 mm bridge width
  const TEMPLE_L = 0.135;                    // 135 mm arm length

  // 1. Lens rings (sleek round rims) -----------------------------------------
  const ringGeo = new THREE.TorusGeometry(LENS_R, LENS_TUBE, 24, 64);
  const ringR = new THREE.Mesh(ringGeo, frameMat);
  ringR.position.set(EYE_GAP / 2, 0, 0);

  const ringL = new THREE.Mesh(ringGeo, frameMat);
  ringL.position.set(-EYE_GAP / 2, 0, 0);
  group.add(ringR, ringL);

  // 2. Optical Tinted Lenses -------------------------------------------------
  const lensGeo = new THREE.CircleGeometry(LENS_R - LENS_TUBE * 0.5, 64);
  const lensR = new THREE.Mesh(lensGeo, lensMat);
  lensR.position.set(EYE_GAP / 2, 0, 0.0004);

  const lensL = new THREE.Mesh(lensGeo, lensMat);
  lensL.position.set(-EYE_GAP / 2, 0, 0.0004);
  group.add(lensR, lensL);

  // 3. Dual Bridge -----------------------------------------------------------
  const bridgeCurve = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(BRIDGE_W / 2 + LENS_TUBE, 0, 0),
    new THREE.Vector3(0, 0.0025, 0.0008),
    new THREE.Vector3(-(BRIDGE_W / 2 + LENS_TUBE), 0, 0)
  );
  const bridgeGeo = new THREE.TubeGeometry(bridgeCurve, 16, 0.0011, 10, false);
  const bridge = new THREE.Mesh(bridgeGeo, frameMat);
  group.add(bridge);

  // Upper brow bar
  const browCurve = new THREE.LineCurve3(
    new THREE.Vector3(BRIDGE_W / 2 + LENS_TUBE, 0.0065, -0.0008),
    new THREE.Vector3(-(BRIDGE_W / 2 + LENS_TUBE), 0.0065, -0.0008)
  );
  const browGeo = new THREE.TubeGeometry(browCurve, 10, 0.0010, 8, false);
  const browBar = new THREE.Mesh(browGeo, frameMat);
  group.add(browBar);

  // 4. Nose Pads & Pad Arms (Rest gently on nasal bridge) --------------------
  const padGeo = new THREE.BoxGeometry(0.0028, 0.006, 0.0016);
  const armCurveR = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(BRIDGE_W / 2, 0.001, 0),
    new THREE.Vector3(0.006, -0.004, -0.0035),
    new THREE.Vector3(0.008, -0.008, -0.0060)
  );
  const armGeoR = new THREE.TubeGeometry(armCurveR, 8, 0.0007, 6, false);
  const padArmR = new THREE.Mesh(armGeoR, goldAccentMat);
  const padR = new THREE.Mesh(padGeo, padMat);
  padR.position.set(0.008, -0.008, -0.0060);
  padR.rotation.set(-0.2, 0.25, -0.15);
  group.add(padArmR, padR);

  const armCurveL = new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(-BRIDGE_W / 2, 0.001, 0),
    new THREE.Vector3(-0.006, -0.004, -0.0035),
    new THREE.Vector3(-0.008, -0.008, -0.0060)
  );
  const armGeoL = new THREE.TubeGeometry(armCurveL, 8, 0.0007, 6, false);
  const padArmL = new THREE.Mesh(armGeoL, goldAccentMat);
  const padL = new THREE.Mesh(padGeo, padMat);
  padL.position.set(-0.008, -0.008, -0.0060);
  padL.rotation.set(-0.2, -0.25, 0.15);
  group.add(padArmL, padL);

  // 5. End pieces & Hinges (Outer connection at frame corners) ----------------
  const hingeGeo = new THREE.BoxGeometry(0.0035, 0.003, 0.005);
  const hingeR = new THREE.Mesh(hingeGeo, goldAccentMat);
  hingeR.position.set(EYE_GAP / 2 + LENS_R + 0.002, 0.002, -0.0025);

  const hingeL = new THREE.Mesh(hingeGeo, goldAccentMat);
  hingeL.position.set(-(EYE_GAP / 2 + LENS_R + 0.002), 0.002, -0.0025);
  group.add(hingeR, hingeL);

  // 6. Straight Horizontal Temple Arms (Run at eye level straight back to ears)
  const templeGeo = new THREE.CylinderGeometry(0.0011, 0.0011, TEMPLE_L, 12);
  templeGeo.rotateX(Math.PI / 2);
  templeGeo.translate(0, 0, -TEMPLE_L / 2);

  const templeR = new THREE.Mesh(templeGeo, frameMat);
  templeR.position.set(EYE_GAP / 2 + LENS_R + 0.003, 0.002, -0.003);

  const templeL = new THREE.Mesh(templeGeo, frameMat);
  templeL.position.set(-(EYE_GAP / 2 + LENS_R + 0.003), 0.002, -0.003);

  group.add(templeR, templeL);

  return group;
}
