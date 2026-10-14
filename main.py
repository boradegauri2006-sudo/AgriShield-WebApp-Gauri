# HISTORY
# =========================================================
elif st.session_state.page == "History":
    st.title(t["history"])
    records = st.session_state.history

    if not records:
        st.info(t["history_empty"])
    else:
        st.caption(
            "History is stored in a local JSON file. On Streamlit Cloud, it may "
            "not survive app restarts and should not be treated as a private "
            "multi-user database."
        )

        if st.button(t["clear"], type="secondary"):
            st.session_state.history = []
            st.session_state.latest_result = None
            saved = save_history([])
            if saved:
                st.success("History cleared.")
            else:
                st.warning("History cleared for this session, but the file could not be updated.")
            st.rerun()

        for i, record in enumerate(records):
            with st.expander(
                f'{record.get("date", "Unknown date")} · '
                f'{record.get("display_name", "Unknown result")} · '
                f'{record.get("confidence", 0):.1f}%'
            ):
                st.write(f'**{t["confidence"]}:** {record.get("confidence", 0):.1f}%')
                st.write(f'**{t["symptoms"]}:**')
                show_bullets(record.get("symptoms", []))
                st.write(f'**{t["actions"]}:**')
                show_bullets(record.get("actions", []))

                pdf = create_pdf_report(record, language)
                if pdf:
                    st.download_button(
                        "Download this report",
                        data=pdf,
                        file_name=f"AgriShield_Report_{i + 1}.pdf",
                        mime="application/pdf",
                        key=f"history_pdf_{i}",
                    )

# =========================================================
# CARE GUIDE
# =========================================================
elif st.session_state.page == "Care":
    st.title(t["care"])
    if language == "मराठी":
        st.markdown('<div class="care-intro"><h3>लहान सवयी, निरोगी झाडे 🌿</h3><p>टोमॅटोच्या झाडांची नियमित काळजी घेतल्याने काही सामान्य समस्यांचा धोका कमी होऊ शकतो.</p></div>', unsafe_allow_html=True)
        care_sections = [
            ("☀️ सूर्यप्रकाश आणि अंतर", ["टोमॅटोच्या झाडांना शक्यतो दररोज ६–८ तास सूर्यप्रकाश मिळू द्या.", "झाडांमध्ये हवा खेळती राहील असे अंतर ठेवा."]),
            ("💧 पाणी देणे", ["झाडाच्या मुळाजवळ पाणी द्या.", "मातीचा ओलावा तपासून पाणी द्या; पाणी साचू देऊ नका.", "पानांवर सतत पाणी राहणे टाळा."]),
            ("🍃 स्वच्छता", ["गळलेली आणि बाधित पाने काढा.", "बाधित झाड हाताळल्यानंतर साधने स्वच्छ करा.", "बाधित अवशेषांची योग्य विल्हेवाट लावा."]),
            ("🔎 नियमित तपासणी", ["पानांच्या वरच्या आणि खालच्या बाजू तपासा.", "पानांवरील डाग, पिवळेपणा किंवा बारीक जाळी पाहा.", "रोग पसरत असल्यास कृषी तज्ज्ञांचा सल्ला घ्या."]),
        ]
    else:
        st.markdown('<div class="care-intro"><h3>Small habits. Healthier plants. 🌿</h3><p>A simple routine can help you notice changes early and keep tomato plants in better growing conditions.</p></div>', unsafe_allow_html=True)
        care_sections = [
            ("☀️ Sunlight and spacing", ["Aim for around 6–8 hours of sunlight when practical.", "Space plants so air can move between them."]),
            ("💧 Watering", ["Water near the base of the plant.", "Check soil moisture and avoid waterlogging.", "Avoid keeping leaves wet for long periods."]),
            ("🍃 Cleanliness", ["Remove fallen and badly affected leaves.", "Clean tools after handling affected plants.", "Dispose of diseased debris appropriately."]),
            ("🔎 Regular inspection", ["Check the upper and lower sides of leaves.", "Look for spots, yellowing or fine webbing.", "Seek expert advice if symptoms spread."]),
        ]
    cols = st.columns(2, gap="large")
    for index, (title, points) in enumerate(care_sections):
        with cols[index % 2]:
            st.markdown(f'<div class="care-panel"><h3>{title}</h3><ul>' + ''.join(f'<li>{point}</li>' for point in points) + '</ul></div>', unsafe_allow_html=True)
    st.info("ही सर्वसाधारण काळजीची माहिती आहे; रोगाचे निदान किंवा उपचारांसाठी कृषी तज्ज्ञांचा सल्ला घ्या." if language == "मराठी" else "These are general care tips, not a diagnosis or treatment plan. Ask a local agriculture expert for disease-specific advice.")

# =========================================================
# ABOUT
# =========================================================
elif st.session_state.page == "About":
    st.title(t["about"])
    if language == "मराठी":
        st.write(
            "AgriShield हा टोमॅटोच्या पानांवरील संभाव्य रोग ओळखण्यासाठी तयार केलेला "
            "विद्यार्थी प्रकल्प आहे. हा CNN आधारित मॉडेल वापरतो. निकाल हा केवळ "
            "प्राथमिक अंदाज आहे; तो तज्ज्ञांच्या निदानाची जागा घेत नाही."
        )
        st.markdown("### वापरलेली तंत्रज्ञाने")
        show_bullets([
            "Python",
            "Streamlit — वेबसाइट इंटरफेस",
            "TensorFlow आणि Keras — प्रशिक्षित मॉडेल",
            "NumPy आणि Pillow — प्रतिमा प्रक्रिया",
            "JSON — स्थानिक इतिहास नोंद",
        ])
    else:
        st.write(
            "AgriShield is a student project that uses a trained CNN model to "
            "estimate which of nine tomato conditions, or a healthy leaf, best "
            "matches an uploaded photo. It is intended as an educational first check."
        )
        st.markdown("### Technologies used")
        show_bullets([
            "Python",
            "Streamlit for the web interface",
            "TensorFlow and Keras for model inference",
            "NumPy and Pillow for image processing",
            "JSON for local history storage",
            "ReportLab for PDF reports",
        ])

    st.warning(
        "Model confidence is not the same as real-world accuracy. The damage "
        "estimate is experimental, may include background pixels, and should "
        "not be used as a treatment decision."
    )

# =========================================================
# FOOTER
# =========================================================
st.markdown(
    '<div class="ag-footer">🍃 <b>AgriShield</b> · Student project · '
    'AI predictions should be confirmed by an agriculture expert.</div>',
    unsafe_allow_html=True,
)
