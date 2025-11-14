#!/bin/bash

# Revolutionary MSA + Templates Offline Prediction Script
# Usage: ./predict.sh "SEQUENCE"
# Features: Dual validation, FOLDCOMP extraction, 90+ plDDT targeting

if [ $# -eq 0 ]; then
    echo "Usage: $0 <protein_sequence>"
    echo "Example: $0 'MKLHYVAVLTLAILMFLTWLPASLSCNKAL'"
    echo ""
    echo "🚀 Revolutionary Features:"
    echo "  ✅ MSA generation with ColabFold"
    echo "  ✅ Dual validation (sequence + 3Di structural)"
    echo "  ✅ FOLDCOMP template extraction"
    echo "  ✅ Biotite PDB→CIF conversion"
    echo "  ✅ Stockholm alignment generation"
    echo "  ✅ Template-enabled OpenFold prediction"
    echo "  🎯 Target: 90+ plDDT accuracy!"
    exit 1
fi

SEQUENCE="$1"
TIMESTAMP=$(date +%s)
OUTPUT_DIR="prediction_${TIMESTAMP}"
QUERY_NAME="query_${TIMESTAMP}"
BASE_QUERY_JSON="base_query_${TIMESTAMP}.json"
TEMPLATE_QUERY_JSON="template_query_${TIMESTAMP}.json"

echo "🚀 REVOLUTIONARY MSA + TEMPLATES PREDICTION PIPELINE"
echo "================================================================="
echo "🧬 Sequence: ${SEQUENCE}"
echo "📁 Output directory: ${OUTPUT_DIR}"
echo "⏰ Timestamp: ${TIMESTAMP}"
echo ""

# Step 1: Create base query JSON
echo "📝 Step 1: Creating base query JSON..."
cat > "${BASE_QUERY_JSON}" << EOF
{
    "seeds": [42],
    "queries": {
        "${QUERY_NAME}": {
            "chains": [
                {
                    "molecule_type": "protein",
                    "chain_ids": ["A"],
                    "sequence": "${SEQUENCE}"
                }
            ]
        }
    }
}
EOF
echo "   ✅ Created: ${BASE_QUERY_JSON}"

# Step 2: Generate MSA using offline pipeline
echo ""
echo "🧬 Step 2: Generating MSA with ColabFold..."
MSA_START_TIME=$(date +%s)
python openfold3/run_openfold.py predict \
    --query_json "${BASE_QUERY_JSON}" \
    --offline_mode true \
    --foldseek_database_dir /Users/gtaghon/foldseek_databases \
    --runner_yaml msa_only_runner.yaml \
    --output_dir "msa_${TIMESTAMP}" \
    --msa_only > "msa_${TIMESTAMP}.log" 2>&1

MSA_EXIT_CODE=$?
MSA_END_TIME=$(date +%s)
MSA_DURATION=$((MSA_END_TIME - MSA_START_TIME))

if [ $MSA_EXIT_CODE -eq 0 ]; then
    echo "   ✅ MSA generation completed (${MSA_DURATION}s)"

    # Find the generated MSA file
    MSA_FILE=$(find /var/folders -name "*main.a3m" -newer "${BASE_QUERY_JSON}" 2>/dev/null | head -1)
    if [ -n "$MSA_FILE" ]; then
        echo "   📄 Found MSA: $MSA_FILE"

        # Step 3: Generate templates using revolutionary dual validation pipeline
        echo ""
        echo "🎯 Step 3: Revolutionary template generation (dual validation)..."
        TEMPLATE_START_TIME=$(date +%s)

        python -c "
import sys
sys.path.append('.')
from benchmarks.msa_to_template_mapper import MSAToTemplateMapper
import json

print('🔥 Initializing revolutionary dual validation pipeline...')
mapper = MSAToTemplateMapper('/Users/gtaghon/foldseek_databases')

print('🧬 Running MSA → Templates pipeline...')
results = mapper.generate_templates_from_msa('$MSA_FILE', max_templates=5)

print(f\"📊 Results: {results.get('count', 0)} templates, mode: {results.get('mode', 'unknown')}\")

if results.get('stockholm') and results.get('count', 0) > 0:
    print('✅ Templates found! Creating template-enabled query...')

    # Load base query
    with open('$BASE_QUERY_JSON', 'r') as f:
        query_data = json.load(f)

    # Add template information
    for query_id, query_info in query_data.get('queries', {}).items():
        for chain in query_info.get('chains', []):
            if chain.get('molecule_type') == 'protein':
                chain['template_alignment_file_path'] = results['stockholm']
                print(f'   ✅ Added template to chain {chain.get(\"chain_ids\", \"unknown\")}')

    # Save template-enabled query
    with open('$TEMPLATE_QUERY_JSON', 'w') as f:
        json.dump(query_data, f, indent=2)

    print(f'📋 Template-enabled query saved: $TEMPLATE_QUERY_JSON')
    print(f'📄 Stockholm file: {results[\"stockholm\"]}')
    print(f'🧬 CIF files: {len(results.get(\"cif_files\", []))}')

    # Save results for later use
    with open('template_results_${TIMESTAMP}.json', 'w') as f:
        json.dump({
            'template_count': results.get('count', 0),
            'stockholm_file': results.get('stockholm'),
            'cif_files': results.get('cif_files', []),
            'mode': results.get('mode', 'unknown')
        }, f, indent=2)

    exit(0)
else:
    print('⚠️ No templates found - will use MSA-only mode')
    import shutil
    shutil.copy('$BASE_QUERY_JSON', '$TEMPLATE_QUERY_JSON')

    with open('template_results_${TIMESTAMP}.json', 'w') as f:
        json.dump({
            'template_count': 0,
            'stockholm_file': None,
            'cif_files': [],
            'mode': 'msa_only'
        }, f, indent=2)

    exit(1)
"

        TEMPLATE_EXIT_CODE=$?
        TEMPLATE_END_TIME=$(date +%s)
        TEMPLATE_DURATION=$((TEMPLATE_END_TIME - TEMPLATE_START_TIME))

        if [ $TEMPLATE_EXIT_CODE -eq 0 ]; then
            echo "   ✅ Template generation completed (${TEMPLATE_DURATION}s)"
            PREDICTION_MODE="MSA + TEMPLATES (targeting 90+ plDDT)"
        else
            echo "   ⚠️ Template generation failed, using MSA-only mode (${TEMPLATE_DURATION}s)"
            PREDICTION_MODE="MSA ONLY (targeting ~60 plDDT)"
        fi
    else
        echo "   ❌ MSA file not found, creating basic query"
        cp "${BASE_QUERY_JSON}" "${TEMPLATE_QUERY_JSON}"
        PREDICTION_MODE="BASIC (no MSA, low accuracy)"
    fi
else
    echo "   ❌ MSA generation failed (${MSA_DURATION}s), using basic query"
    cp "${BASE_QUERY_JSON}" "${TEMPLATE_QUERY_JSON}"
    PREDICTION_MODE="BASIC (no MSA, low accuracy)"
fi

# Step 4: Final OpenFold prediction with templates
echo ""
echo "🎯 Step 4: Final OpenFold prediction..."
echo "   🧬 Mode: ${PREDICTION_MODE}"
echo "   📋 Query: ${TEMPLATE_QUERY_JSON}"
echo "   📁 Output: ${OUTPUT_DIR}"
echo ""

PREDICTION_START_TIME=$(date +%s)
python openfold3/run_openfold.py predict \
    --query_json "${TEMPLATE_QUERY_JSON}" \
    --offline_mode true \
    --foldseek_database_dir /Users/gtaghon/foldseek_databases \
    --runner_yaml offline_runner_config.yaml \
    --output_dir "${OUTPUT_DIR}"

PREDICTION_EXIT_CODE=$?
PREDICTION_END_TIME=$(date +%s)
PREDICTION_DURATION=$((PREDICTION_END_TIME - PREDICTION_START_TIME))
TOTAL_DURATION=$((PREDICTION_END_TIME - MSA_START_TIME))

echo ""
echo "🎉 PREDICTION COMPLETED!"
echo "================================================================="
echo "⏱️  Timing:"
echo "   MSA generation: ${MSA_DURATION}s"
echo "   Template generation: ${TEMPLATE_DURATION}s"
echo "   Final prediction: ${PREDICTION_DURATION}s"
echo "   Total runtime: ${TOTAL_DURATION}s"
echo ""

# Analyze results
echo "📊 RESULTS ANALYSIS:"
if [ $PREDICTION_EXIT_CODE -eq 0 ] && [ -d "${OUTPUT_DIR}/${QUERY_NAME}/seed_42" ]; then
    FOUND_RESULTS=false
    for conf_file in "${OUTPUT_DIR}/${QUERY_NAME}/seed_42"/*_confidences_aggregated.json; do
        if [ -f "$conf_file" ]; then
            sample=$(basename "$conf_file" | sed 's/.*_sample_\([0-9]*\)_.*/\1/')
            plddt=$(grep -o '"avg_plddt": [0-9.]*' "$conf_file" | cut -d' ' -f2 | head -1)

            if [ -n "$plddt" ]; then
                echo "   🎯 Sample ${sample}: plDDT = ${plddt}"

                # Check if we achieved our target
                if (( $(echo "$plddt >= 90" | bc -l 2>/dev/null || echo "0") )); then
                    echo "      🎉 BREAKTHROUGH! Achieved 90+ plDDT target!"
                elif (( $(echo "$plddt >= 80" | bc -l 2>/dev/null || echo "0") )); then
                    echo "      ✅ Excellent accuracy (80+ plDDT)"
                elif (( $(echo "$plddt >= 70" | bc -l 2>/dev/null || echo "0") )); then
                    echo "      👍 Good accuracy (70+ plDDT)"
                elif (( $(echo "$plddt >= 60" | bc -l 2>/dev/null || echo "0") )); then
                    echo "      📊 Moderate accuracy (60+ plDDT)"
                else
                    echo "      ⚠️ Low accuracy (<60 plDDT)"
                fi
                FOUND_RESULTS=true
            fi
        fi
    done

    if [ "$FOUND_RESULTS" = false ]; then
        echo "   ❌ No confidence scores found"
    fi
else
    echo "   ❌ Prediction failed or no results generated"
fi

# Show template information if available
if [ -f "template_results_${TIMESTAMP}.json" ]; then
    TEMPLATE_COUNT=$(grep -o '"template_count": [0-9]*' "template_results_${TIMESTAMP}.json" | cut -d' ' -f2 2>/dev/null || echo "0")
    echo ""
    echo "🏗️ Template Information:"
    echo "   Templates used: ${TEMPLATE_COUNT}"
    if [ "$TEMPLATE_COUNT" -gt 0 ]; then
        echo "   ✅ Revolutionary dual validation successful!"
        echo "   ✅ MSA + Templates pipeline used"
    else
        echo "   ⚠️ MSA-only mode used"
    fi
fi

echo ""
echo "📁 Generated Files:"
echo "   - Base query: ${BASE_QUERY_JSON}"
echo "   - Template query: ${TEMPLATE_QUERY_JSON}"
echo "   - Results: ${OUTPUT_DIR}/"
echo "   - MSA log: msa_${TIMESTAMP}.log"
if [ -f "template_results_${TIMESTAMP}.json" ]; then
    echo "   - Template info: template_results_${TIMESTAMP}.json"
fi

echo ""
echo "🔍 Debug Commands:"
echo "   View MSA log: cat msa_${TIMESTAMP}.log"
echo "   View results: ls -la ${OUTPUT_DIR}/${QUERY_NAME}/seed_42/"
echo "   View template info: cat template_results_${TIMESTAMP}.json"

if [ $PREDICTION_EXIT_CODE -eq 0 ]; then
    echo ""
    echo "🎉 SUCCESS! Revolutionary MSA + Templates pipeline completed!"
else
    echo ""
    echo "⚠️ Prediction completed with warnings - check debug info above"
fi